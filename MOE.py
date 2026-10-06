import os
import time
import random
import collections
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from PIL import Image
from carla_env import CarlaEnv
from yolo import YOLO
from transformers import DPTForDepthEstimation, DPTFeatureExtractor

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
DPT_MODEL_PATH = os.environ.get("DPT_MODEL_PATH", "/root/autodl-fs/Uni yolo/dpt-large")
DPT_CALIBRATION_PATH = os.environ.get("DPT_METRIC_CALIBRATION", "/root/autodl-fs/Uni yolo/dpt_metric_calibration.npz")
TOP_K = 5
N_THRESHOLD = 5
DELTA = 0.1
LANE_WIDTH = 3.5

RISK_RULES = {
    "vehicle": (0.7, 0.2, 15.0),
    "person": (1.0, 0.7, 12.0),
    "obstacle": (1.0, 0.5, 10.0),
    "traffic_sign": (0.5, 0.5, None),
    "unknown": (0.8, 0.5, 12.0),
}

OBSTACLE_CLASSES = {
    "traffic cone", "barrier", "roadblock", "debris", "broken vehicle", "animal",
    "dog", "cat", "horse", "traffic barrel", "mailbox", "bench", "bus stop shelter"
}
PERSON_CLASSES = {
    "person", "child", "pedestrian with stroller", "pedestrian with luggage",
    "pedestrian with umbrella", "pedestrian with shopping cart"
}
VEHICLE_CLASSES = {
    "car", "truck", "bus", "motorbike", "motorcycle", "bicycle", "van", "taxi",
    "vehicle", "ambulance", "fire truck", "police car", "trailer", "pickup", "suv",
    "minibus", "sports car"
}
TRAFFIC_SIGN_NUMBERS = {"30", "40", "50", "60", "70", "80", "90", "100", "110", "120"}
TRAFFIC_SIGN_KEYWORDS = {"traffic sign", "traffic light"}
WORDS_TO_NUM = {
    "thirty": "30", "forty": "40", "fifty": "50", "sixty": "60", "seventy": "70",
    "eighty": "80", "ninety": "90", "one hundred": "100", "one hundred ten": "110",
    "one hundred twenty": "120"
}


def map_class_id(original_class, confidence):
    if confidence < 0.20:
        return "unknown"
    if original_class is None:
        return "unknown"
    cls = WORDS_TO_NUM.get(str(original_class).strip().lower(), str(original_class).strip().lower())
    if cls in PERSON_CLASSES:
        return "person"
    if cls in OBSTACLE_CLASSES:
        return "obstacle"
    if cls in VEHICLE_CLASSES:
        return "vehicle"
    if cls in TRAFFIC_SIGN_KEYWORDS or cls in TRAFFIC_SIGN_NUMBERS:
        return "traffic_sign"
    return "unknown"


def calculate_risk_value(class_id, distance):
    near_risk, far_risk, threshold = RISK_RULES.get(class_id, RISK_RULES["unknown"])
    if threshold is None:
        return near_risk
    return near_risk if distance <= threshold else far_risk


def calculate_weight(class_id, distance):
    weight = 1.0
    if class_id != "traffic_sign" and distance <= 10.0:
        weight *= 1.5
    if class_id == "unknown":
        weight *= 1.2
    if class_id == "traffic_sign":
        weight *= 1.1
    return weight


def calculate_emotion_score(detections, top_k=TOP_K, n_threshold=N_THRESHOLD, delta=DELTA):
    if not detections:
        return 0.0
    scored = []
    for detection in detections:
        class_id = detection["class_id"]
        distance = float(detection["distance"])
        risk = calculate_risk_value(class_id, distance)
        weight = calculate_weight(class_id, distance)
        scored.append((weight * risk, weight))
    n_detected = len(scored)
    scored.sort(key=lambda item: item[0], reverse=True)
    focused = scored[:min(top_k, n_detected)]
    f_num = 1.0 if n_detected <= n_threshold else 1.0 + delta * (n_detected - n_threshold)
    numerator = sum(item[0] for item in focused)
    denominator = sum(item[1] for item in focused) * f_num
    return 0.0 if denominator <= 0.0 else float(numerator / denominator)


def emotion_to_expert(emotion_score):
    if emotion_score >= 0.7:
        return "DDPG"
    if emotion_score >= 0.4:
        return "SAC"
    return "PPO"


def calculate_risk_cost(state, collision_indicator, lane_width=LANE_WIDTH):
    state = np.asarray(state, dtype=np.float32)
    if state.shape[0] < 13:
        raise ValueError("State vector must contain at least 13 elements to evaluate the paper-defined risk cost.")
    y = float(state[8])
    y_cur = float(state[9])
    y_tar = float(state[10])
    d_min = float(state[12])
    y_min = min(y_cur, y_tar)
    y_max = max(y_cur, y_tar)
    y_bar = min(max(y, y_min), y_max)
    zeta_uld = min(1.0, abs(y - y_bar) / lane_width)
    proximity = max(0.0, 5.0 - d_min) / 5.0
    return float(collision_indicator) + 0.2 * zeta_uld + 0.8 * proximity


def extract_collision_indicator(info):
    for key in ("collision", "collision_occurred", "is_collision", "has_collision"):
        if key in info:
            return float(bool(info[key]))
    return None


class DPTMetricCalibrator:
    def __init__(self, calibration_path, required_frames=12, min_depth=1.5, max_depth=50.0, samples_per_frame=25000):
        self.calibration_path = calibration_path
        self.required_frames = int(required_frames)
        self.min_depth = float(min_depth)
        self.max_depth = float(max_depth)
        self.samples_per_frame = int(samples_per_frame)
        self.scale = None
        self.shift = None
        self.raw_samples = []
        self.metric_inv_samples = []
        self.frames_collected = 0
        self.rng = np.random.default_rng(0)
        if os.path.isfile(calibration_path):
            params = np.load(calibration_path)
            if "scale" in params and "shift" in params:
                self.scale = float(params["scale"])
                self.shift = float(params["shift"])

    @property
    def ready(self):
        return self.scale is not None and self.shift is not None

    def _resize_metric_depth(self, metric_depth, target_shape):
        metric_depth = np.asarray(metric_depth, dtype=np.float32)
        if metric_depth.shape == target_shape:
            return metric_depth
        depth_t = torch.as_tensor(metric_depth, dtype=torch.float32, device=DEVICE)[None, None]
        resized = F.interpolate(depth_t, size=target_shape, mode="nearest")
        return resized[0, 0].detach().cpu().numpy()

    def _decode_carla_depth(self, depth_source):
        if depth_source is None:
            return None
        if isinstance(depth_source, np.ndarray):
            if depth_source.ndim == 2:
                return depth_source.astype(np.float32)
            if depth_source.ndim == 3 and depth_source.shape[2] >= 3:
                array = depth_source.astype(np.uint32)
                if depth_source.dtype == np.uint8:
                    b = array[:, :, 0]
                    g = array[:, :, 1]
                    r = array[:, :, 2]
                    normalized = (r + 256 * g + 65536 * b) / 16777215.0
                    return (1000.0 * normalized).astype(np.float32)
        if isinstance(depth_source, Image.Image):
            array = np.asarray(depth_source.convert("RGB"), dtype=np.uint32)
            r = array[:, :, 0]
            g = array[:, :, 1]
            b = array[:, :, 2]
            normalized = (r + 256 * g + 65536 * b) / 16777215.0
            return (1000.0 * normalized).astype(np.float32)
        if hasattr(depth_source, "raw_data") and hasattr(depth_source, "width") and hasattr(depth_source, "height"):
            array = np.frombuffer(depth_source.raw_data, dtype=np.uint8)
            array = array.reshape((int(depth_source.height), int(depth_source.width), 4))
            b = array[:, :, 0].astype(np.uint32)
            g = array[:, :, 1].astype(np.uint32)
            r = array[:, :, 2].astype(np.uint32)
            normalized = (r + 256 * g + 65536 * b) / 16777215.0
            return (1000.0 * normalized).astype(np.float32)
        return None

    def metric_reference_from_source(self, source):
        for name in (
            "current_depth_map",
            "metric_depth_map",
            "current_metric_depth",
            "current_depth_image",
            "depth_image",
            "current_depth",
        ):
            if hasattr(source, name):
                metric_depth = self._decode_carla_depth(getattr(source, name))
                if metric_depth is not None:
                    return metric_depth
        return None

    def add_frame(self, raw_inverse_depth, metric_depth):
        raw_inverse_depth = np.asarray(raw_inverse_depth, dtype=np.float32)
        metric_depth = self._resize_metric_depth(metric_depth, raw_inverse_depth.shape)
        valid = (
            np.isfinite(raw_inverse_depth)
            & np.isfinite(metric_depth)
            & (raw_inverse_depth > 0.0)
            & (metric_depth >= self.min_depth)
            & (metric_depth <= self.max_depth)
        )
        indices = np.flatnonzero(valid)
        if indices.size < 2000:
            raise RuntimeError("Insufficient valid CARLA metric-depth pixels for DPT calibration.")
        if indices.size > self.samples_per_frame:
            indices = self.rng.choice(indices, self.samples_per_frame, replace=False)
        raw_flat = raw_inverse_depth.reshape(-1)[indices].astype(np.float64)
        metric_flat = metric_depth.reshape(-1)[indices].astype(np.float64)
        self.raw_samples.append(raw_flat)
        self.metric_inv_samples.append(1.0 / metric_flat)
        self.frames_collected += 1
        if self.frames_collected >= self.required_frames:
            self.fit()

    def fit(self):
        x = np.concatenate(self.raw_samples)
        y = np.concatenate(self.metric_inv_samples)
        keep = np.isfinite(x) & np.isfinite(y)
        x = x[keep]
        y = y[keep]
        if x.size < 5000:
            raise RuntimeError("Insufficient calibration samples for DPT metric alignment.")
        for _ in range(6):
            A = np.column_stack((x, np.ones_like(x)))
            scale, shift = np.linalg.lstsq(A, y, rcond=None)[0]
            residual = y - (scale * x + shift)
            median = np.median(residual)
            mad = np.median(np.abs(residual - median)) + 1e-12
            robust_sigma = 1.4826 * mad
            next_keep = np.abs(residual - median) <= 3.0 * robust_sigma
            if next_keep.sum() < 5000 or next_keep.sum() == x.size:
                break
            x = x[next_keep]
            y = y[next_keep]
        if not np.isfinite(scale) or not np.isfinite(shift) or scale <= 0.0:
            raise RuntimeError("Invalid DPT metric calibration parameters.")
        self.scale = float(scale)
        self.shift = float(shift)
        directory = os.path.dirname(self.calibration_path)
        if directory:
            os.makedirs(directory, exist_ok=True)
        np.savez(
            self.calibration_path,
            scale=np.float32(self.scale),
            shift=np.float32(self.shift),
            min_depth=np.float32(self.min_depth),
            max_depth=np.float32(self.max_depth),
            frames=np.int32(self.frames_collected),
        )
        self.raw_samples.clear()
        self.metric_inv_samples.clear()

    def transform(self, raw_inverse_depth):
        if not self.ready:
            raise RuntimeError("DPT metric calibration has not been completed.")
        inverse_metric_depth = self.scale * np.asarray(raw_inverse_depth, dtype=np.float32) + self.shift
        metric_depth = np.full_like(inverse_metric_depth, np.inf, dtype=np.float32)
        valid = inverse_metric_depth > 1e-6
        metric_depth[valid] = 1.0 / inverse_metric_depth[valid]
        return np.clip(metric_depth, 0.0, 1000.0)


class DetectorWithDepth:
    def __init__(self, model_path=DPT_MODEL_PATH, calibration_path=DPT_CALIBRATION_PATH):
        self.yolo = YOLO()
        self.depth_model = DPTForDepthEstimation.from_pretrained(model_path).to(DEVICE).eval()
        self.depth_feature_extractor = DPTFeatureExtractor.from_pretrained(model_path)
        self.metric_calibrator = DPTMetricCalibrator(calibration_path)

    def get_camera_image(self, source):
        camera_img = getattr(source, "current_image", None)
        if camera_img is None:
            return Image.new("RGB", (640, 480))
        return camera_img

    def estimate_relative_inverse_depth(self, pil_image):
        image = pil_image.convert("RGB")
        width, height = image.size
        inputs = self.depth_feature_extractor(images=image, return_tensors="pt")
        inputs = {key: value.to(DEVICE) for key, value in inputs.items()}
        with torch.inference_mode():
            outputs = self.depth_model(**inputs)
            predicted = outputs.predicted_depth.unsqueeze(1)
            predicted = F.interpolate(predicted, size=(height, width), mode="bicubic", align_corners=False)
        return predicted[0, 0].detach().cpu().numpy().astype(np.float32)

    def collect_metric_calibration(self, source):
        camera_img = self.get_camera_image(source)
        raw_inverse_depth = self.estimate_relative_inverse_depth(camera_img)
        metric_depth = self.metric_calibrator.metric_reference_from_source(source)
        if metric_depth is None:
            raise RuntimeError(
                "A synchronized CARLA metric-depth frame is required only for the one-time DPT calibration. "
                "Expose it as current_depth_map, metric_depth_map, current_metric_depth, current_depth_image, depth_image, or current_depth."
            )
        self.metric_calibrator.add_frame(raw_inverse_depth, metric_depth)

    def estimate_depth(self, pil_image):
        raw_inverse_depth = self.estimate_relative_inverse_depth(pil_image)
        return self.metric_calibrator.transform(raw_inverse_depth)

    def _object_distance(self, depth_map, x1, y1, x2, y2):
        box_width = x2 - x1
        box_height = y2 - y1
        mx = max(1, int(round(box_width * 0.20)))
        my = max(1, int(round(box_height * 0.20)))
        rx1 = min(x2 - 1, x1 + mx)
        rx2 = max(rx1 + 1, x2 - mx)
        ry1 = min(y2 - 1, y1 + my)
        ry2 = max(ry1 + 1, y2 - my)
        region = depth_map[ry1:ry2, rx1:rx2]
        finite = region[np.isfinite(region) & (region > 0.0)]
        if finite.size == 0:
            region = depth_map[y1:y2, x1:x2]
            finite = region[np.isfinite(region) & (region > 0.0)]
        if finite.size == 0:
            return None
        q1, q3 = np.percentile(finite, [25.0, 75.0])
        trimmed = finite[(finite >= q1) & (finite <= q3)]
        if trimmed.size == 0:
            trimmed = finite
        return float(np.median(trimmed))

    def detect_objects(self, source):
        camera_img = self.get_camera_image(source)
        _, yolo_dets = self.yolo.uni_detect_image(camera_img, crop=True, count=False)
        depth_map = self.estimate_depth(camera_img)
        height, width = depth_map.shape
        results = []
        for det in yolo_dets:
            cls_raw = det.get("class_id") or det.get("class") or det.get("name") or det.get("label")
            conf = det.get("confidence")
            if conf is None:
                conf = det.get("score")
            if conf is None:
                conf = det.get("conf")
            conf = float(conf) if conf is not None else 0.0
            box = det.get("box", det.get("bbox", [0, 0, 0, 0]))
            if box is None or len(box) != 4:
                continue
            x1, y1, x2, y2 = map(int, box)
            x1 = max(0, min(width - 1, x1))
            x2 = max(0, min(width, x2))
            y1 = max(0, min(height - 1, y1))
            y2 = max(0, min(height, y2))
            if x2 <= x1 or y2 <= y1:
                continue
            distance = self._object_distance(depth_map, x1, y1, x2, y2)
            if distance is None:
                continue
            mapped_cls = map_class_id(cls_raw, conf)
            results.append({"class_id": mapped_cls, "confidence": conf, "distance": distance})
        return results


def calibrate_dpt_metric(env, detector, frames=12):
    if detector.metric_calibrator.ready:
        return
    detector.metric_calibrator.required_frames = int(frames)
    env.reset()
    zero_action = np.zeros(env.action_space.shape[0], dtype=np.float32)
    while not detector.metric_calibrator.ready:
        detector.collect_metric_calibration(env)
        if detector.metric_calibrator.ready:
            break
        _, _, done, _ = env.step(zero_action, "PPO")
        if done:
            env.reset()
    env.reset()


class ReplayBuffer:
    def __init__(self, capacity=100000):
        self.buffer = collections.deque(maxlen=capacity)

    def push(self, state, action, reward, next_state, done, cost):
        self.buffer.append((state, action, reward, next_state, done, cost))

    def sample(self, batch_size):
        batch = random.sample(self.buffer, batch_size)
        states, actions, rewards, next_states, dones, costs = zip(*batch)
        return (
            np.asarray(states, dtype=np.float32),
            np.asarray(actions, dtype=np.float32),
            np.asarray(rewards, dtype=np.float32),
            np.asarray(next_states, dtype=np.float32),
            np.asarray(dones, dtype=np.float32),
            np.asarray(costs, dtype=np.float32),
        )

    def __len__(self):
        return len(self.buffer)


class StateActionNetwork(nn.Module):
    def __init__(self, state_dim, action_dim, hidden_dim=256, nonnegative=False):
        super().__init__()
        self.fc1 = nn.Linear(state_dim + action_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.fc3 = nn.Linear(hidden_dim, 1)
        self.nonnegative = nonnegative

    def forward(self, state, action):
        x = torch.cat([state, action], dim=1)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return F.softplus(x) if self.nonnegative else x


class ActorDDPG(nn.Module):
    def __init__(self, state_dim, action_dim, hidden_dim=256, max_action=1.0):
        super().__init__()
        self.fc1 = nn.Linear(state_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.fc3 = nn.Linear(hidden_dim, action_dim)
        self.max_action = max_action

    def forward(self, state):
        x = F.relu(self.fc1(state))
        x = F.relu(self.fc2(x))
        return self.max_action * torch.tanh(self.fc3(x))


class DDPGAgent:
    def __init__(self, state_dim, action_dim, max_action=1.0, gamma=0.99, tau=0.005, lr=1e-3, hidden_dim=256):
        self.actor = ActorDDPG(state_dim, action_dim, hidden_dim, max_action).to(DEVICE)
        self.actor_target = ActorDDPG(state_dim, action_dim, hidden_dim, max_action).to(DEVICE)
        self.actor_target.load_state_dict(self.actor.state_dict())
        self.critic = StateActionNetwork(state_dim, action_dim, hidden_dim).to(DEVICE)
        self.critic_target = StateActionNetwork(state_dim, action_dim, hidden_dim).to(DEVICE)
        self.critic_target.load_state_dict(self.critic.state_dict())
        self.risk_model = StateActionNetwork(state_dim, action_dim, hidden_dim, nonnegative=True).to(DEVICE)
        self.actor_optimizer = optim.Adam(self.actor.parameters(), lr=lr)
        self.critic_optimizer = optim.Adam(self.critic.parameters(), lr=lr)
        self.risk_optimizer = optim.Adam(self.risk_model.parameters(), lr=lr)
        self.gamma = gamma
        self.tau = tau
        self.replay_buffer = ReplayBuffer()

    def select_action(self, state):
        state_t = torch.as_tensor(state, dtype=torch.float32, device=DEVICE).unsqueeze(0)
        with torch.no_grad():
            return self.actor(state_t).cpu().numpy()[0]

    def update(self, batch_size=64, lambda_risk=0.0):
        if len(self.replay_buffer) < batch_size:
            return
        states, actions, rewards, next_states, dones, costs = self.replay_buffer.sample(batch_size)
        states = torch.as_tensor(states, dtype=torch.float32, device=DEVICE)
        actions = torch.as_tensor(actions, dtype=torch.float32, device=DEVICE)
        rewards = torch.as_tensor(rewards, dtype=torch.float32, device=DEVICE).unsqueeze(1)
        next_states = torch.as_tensor(next_states, dtype=torch.float32, device=DEVICE)
        dones = torch.as_tensor(dones, dtype=torch.float32, device=DEVICE).unsqueeze(1)
        costs = torch.as_tensor(costs, dtype=torch.float32, device=DEVICE).unsqueeze(1)
        with torch.no_grad():
            next_actions = self.actor_target(next_states)
            target_q = rewards + (1.0 - dones) * self.gamma * self.critic_target(next_states, next_actions)
        critic_loss = F.mse_loss(self.critic(states, actions), target_q)
        self.critic_optimizer.zero_grad(set_to_none=True)
        critic_loss.backward()
        self.critic_optimizer.step()
        risk_loss = F.mse_loss(self.risk_model(states, actions), costs)
        self.risk_optimizer.zero_grad(set_to_none=True)
        risk_loss.backward()
        self.risk_optimizer.step()
        for parameter in self.critic.parameters():
            parameter.requires_grad_(False)
        for parameter in self.risk_model.parameters():
            parameter.requires_grad_(False)
        actor_actions = self.actor(states)
        actor_loss = -self.critic(states, actor_actions).mean() + lambda_risk * self.risk_model(states, actor_actions).mean()
        self.actor_optimizer.zero_grad(set_to_none=True)
        actor_loss.backward()
        self.actor_optimizer.step()
        for parameter in self.critic.parameters():
            parameter.requires_grad_(True)
        for parameter in self.risk_model.parameters():
            parameter.requires_grad_(True)
        self._soft_update(self.critic, self.critic_target)
        self._soft_update(self.actor, self.actor_target)

    def _soft_update(self, source, target):
        with torch.no_grad():
            for source_param, target_param in zip(source.parameters(), target.parameters()):
                target_param.mul_(1.0 - self.tau).add_(self.tau * source_param)

    def store(self, state, action, reward, next_state, done, cost):
        self.replay_buffer.push(state, action, reward, next_state, done, cost)


class ActorSAC(nn.Module):
    def __init__(self, state_dim, action_dim, hidden_dim=256, max_action=1.0, log_std_min=-20, log_std_max=2):
        super().__init__()
        self.fc1 = nn.Linear(state_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.mean_linear = nn.Linear(hidden_dim, action_dim)
        self.log_std_linear = nn.Linear(hidden_dim, action_dim)
        self.log_std_min = log_std_min
        self.log_std_max = log_std_max
        self.max_action = max_action

    def forward(self, state):
        x = F.relu(self.fc1(state))
        x = F.relu(self.fc2(x))
        mean = self.mean_linear(x)
        log_std = torch.clamp(self.log_std_linear(x), self.log_std_min, self.log_std_max)
        return mean, log_std

    def sample(self, state):
        mean, log_std = self.forward(state)
        std = log_std.exp()
        normal = torch.distributions.Normal(mean, std)
        z = normal.rsample()
        squashed = torch.tanh(z)
        action = squashed * self.max_action
        log_prob = normal.log_prob(z) - torch.log(self.max_action * (1.0 - squashed.pow(2)) + 1e-7)
        return action, log_prob.sum(dim=1, keepdim=True)


class SACAgent:
    def __init__(self, state_dim, action_dim, max_action=1.0, gamma=0.99, tau=0.005, alpha=0.2, lr=3e-4, hidden_dim=256):
        self.actor = ActorSAC(state_dim, action_dim, hidden_dim, max_action).to(DEVICE)
        self.critic1 = StateActionNetwork(state_dim, action_dim, hidden_dim).to(DEVICE)
        self.critic2 = StateActionNetwork(state_dim, action_dim, hidden_dim).to(DEVICE)
        self.critic1_target = StateActionNetwork(state_dim, action_dim, hidden_dim).to(DEVICE)
        self.critic2_target = StateActionNetwork(state_dim, action_dim, hidden_dim).to(DEVICE)
        self.critic1_target.load_state_dict(self.critic1.state_dict())
        self.critic2_target.load_state_dict(self.critic2.state_dict())
        self.risk_model = StateActionNetwork(state_dim, action_dim, hidden_dim, nonnegative=True).to(DEVICE)
        self.actor_optimizer = optim.Adam(self.actor.parameters(), lr=lr)
        self.critic1_optimizer = optim.Adam(self.critic1.parameters(), lr=lr)
        self.critic2_optimizer = optim.Adam(self.critic2.parameters(), lr=lr)
        self.risk_optimizer = optim.Adam(self.risk_model.parameters(), lr=lr)
        self.gamma = gamma
        self.tau = tau
        self.alpha = alpha
        self.replay_buffer = ReplayBuffer()

    def select_action(self, state):
        state_t = torch.as_tensor(state, dtype=torch.float32, device=DEVICE).unsqueeze(0)
        with torch.no_grad():
            action, _ = self.actor.sample(state_t)
        return action.cpu().numpy()[0]

    def update(self, batch_size=64, lambda_risk=0.0):
        if len(self.replay_buffer) < batch_size:
            return
        states, actions, rewards, next_states, dones, costs = self.replay_buffer.sample(batch_size)
        states = torch.as_tensor(states, dtype=torch.float32, device=DEVICE)
        actions = torch.as_tensor(actions, dtype=torch.float32, device=DEVICE)
        rewards = torch.as_tensor(rewards, dtype=torch.float32, device=DEVICE).unsqueeze(1)
        next_states = torch.as_tensor(next_states, dtype=torch.float32, device=DEVICE)
        dones = torch.as_tensor(dones, dtype=torch.float32, device=DEVICE).unsqueeze(1)
        costs = torch.as_tensor(costs, dtype=torch.float32, device=DEVICE).unsqueeze(1)
        with torch.no_grad():
            next_action, next_log_prob = self.actor.sample(next_states)
            q_next = torch.min(
                self.critic1_target(next_states, next_action),
                self.critic2_target(next_states, next_action),
            ) - self.alpha * next_log_prob
            q_target = rewards + (1.0 - dones) * self.gamma * q_next
        critic1_loss = F.mse_loss(self.critic1(states, actions), q_target)
        critic2_loss = F.mse_loss(self.critic2(states, actions), q_target)
        self.critic1_optimizer.zero_grad(set_to_none=True)
        critic1_loss.backward()
        self.critic1_optimizer.step()
        self.critic2_optimizer.zero_grad(set_to_none=True)
        critic2_loss.backward()
        self.critic2_optimizer.step()
        risk_loss = F.mse_loss(self.risk_model(states, actions), costs)
        self.risk_optimizer.zero_grad(set_to_none=True)
        risk_loss.backward()
        self.risk_optimizer.step()
        for network in (self.critic1, self.critic2, self.risk_model):
            for parameter in network.parameters():
                parameter.requires_grad_(False)
        action_sample, log_prob = self.actor.sample(states)
        q_pi = torch.min(self.critic1(states, action_sample), self.critic2(states, action_sample))
        actor_loss = (self.alpha * log_prob - q_pi).mean() + lambda_risk * self.risk_model(states, action_sample).mean()
        self.actor_optimizer.zero_grad(set_to_none=True)
        actor_loss.backward()
        self.actor_optimizer.step()
        for network in (self.critic1, self.critic2, self.risk_model):
            for parameter in network.parameters():
                parameter.requires_grad_(True)
        self._soft_update(self.critic1, self.critic1_target)
        self._soft_update(self.critic2, self.critic2_target)

    def _soft_update(self, source, target):
        with torch.no_grad():
            for source_param, target_param in zip(source.parameters(), target.parameters()):
                target_param.mul_(1.0 - self.tau).add_(self.tau * source_param)

    def store(self, state, action, reward, next_state, done, cost):
        self.replay_buffer.push(state, action, reward, next_state, done, cost)


class ActorPPO(nn.Module):
    def __init__(self, state_dim, action_dim, hidden_dim=256, max_action=1.0):
        super().__init__()
        self.fc1 = nn.Linear(state_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.mu_head = nn.Linear(hidden_dim, action_dim)
        self.log_std_head = nn.Linear(hidden_dim, action_dim)
        self.max_action = max_action

    def forward(self, state):
        x = F.relu(self.fc1(state))
        x = F.relu(self.fc2(x))
        return self.mu_head(x), torch.clamp(self.log_std_head(x), -20, 2)

    def distribution(self, state):
        mean, log_std = self.forward(state)
        return torch.distributions.Normal(mean, log_std.exp())

    def sample(self, state):
        dist = self.distribution(state)
        z = dist.rsample()
        squashed = torch.tanh(z)
        action = squashed * self.max_action
        log_prob = dist.log_prob(z) - torch.log(self.max_action * (1.0 - squashed.pow(2)) + 1e-7)
        return action, log_prob.sum(dim=1, keepdim=True)

    def log_prob(self, state, action):
        dist = self.distribution(state)
        squashed = torch.clamp(action / self.max_action, -0.999999, 0.999999)
        z = torch.atanh(squashed)
        log_prob = dist.log_prob(z) - torch.log(self.max_action * (1.0 - squashed.pow(2)) + 1e-7)
        return log_prob.sum(dim=1, keepdim=True)


class CriticPPO(nn.Module):
    def __init__(self, state_dim, hidden_dim=256):
        super().__init__()
        self.fc1 = nn.Linear(state_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.v_head = nn.Linear(hidden_dim, 1)

    def forward(self, state):
        x = F.relu(self.fc1(state))
        x = F.relu(self.fc2(x))
        return self.v_head(x)


class PPOAgent:
    def __init__(self, state_dim, action_dim, max_action=1.0, gamma=0.99, lam=0.95, lr=3e-4, eps_clip=0.2, k_epochs=10, hidden_dim=256):
        self.actor = ActorPPO(state_dim, action_dim, hidden_dim, max_action).to(DEVICE)
        self.critic = CriticPPO(state_dim, hidden_dim).to(DEVICE)
        self.risk_model = StateActionNetwork(state_dim, action_dim, hidden_dim, nonnegative=True).to(DEVICE)
        self.actor_optimizer = optim.Adam(self.actor.parameters(), lr=lr)
        self.critic_optimizer = optim.Adam(self.critic.parameters(), lr=lr)
        self.risk_optimizer = optim.Adam(self.risk_model.parameters(), lr=lr)
        self.gamma = gamma
        self.lam = lam
        self.eps_clip = eps_clip
        self.k_epochs = k_epochs
        self.memory = []

    def select_action(self, state):
        state_t = torch.as_tensor(state, dtype=torch.float32, device=DEVICE).unsqueeze(0)
        with torch.no_grad():
            action, log_prob = self.actor.sample(state_t)
        return action.cpu().numpy()[0], float(log_prob.item())

    def store(self, transition):
        state = np.asarray(transition[0], dtype=np.float32)
        if np.isfinite(state).all():
            self.memory.append(transition)

    def update(self, lambda_risk=0.0):
        if not self.memory:
            return
        training_data = self._prepare_training_data()
        states = torch.as_tensor(np.asarray([item[0] for item in training_data]), dtype=torch.float32, device=DEVICE)
        actions = torch.as_tensor(np.asarray([item[1] for item in training_data]), dtype=torch.float32, device=DEVICE)
        old_log_probs = torch.as_tensor(np.asarray([item[2] for item in training_data]), dtype=torch.float32, device=DEVICE).unsqueeze(1)
        returns = torch.as_tensor(np.asarray([item[3] for item in training_data]), dtype=torch.float32, device=DEVICE).unsqueeze(1)
        advantages = torch.as_tensor(np.asarray([item[4] for item in training_data]), dtype=torch.float32, device=DEVICE).unsqueeze(1)
        costs = torch.as_tensor(np.asarray([item[5] for item in training_data]), dtype=torch.float32, device=DEVICE).unsqueeze(1)
        advantages = (advantages - advantages.mean()) / (advantages.std(unbiased=False) + 1e-8)
        for _ in range(self.k_epochs):
            risk_loss = F.mse_loss(self.risk_model(states, actions), costs)
            self.risk_optimizer.zero_grad(set_to_none=True)
            risk_loss.backward()
            self.risk_optimizer.step()
            for parameter in self.risk_model.parameters():
                parameter.requires_grad_(False)
            new_log_probs = self.actor.log_prob(states, actions)
            ratio = torch.exp(new_log_probs - old_log_probs)
            surr1 = ratio * advantages
            surr2 = torch.clamp(ratio, 1.0 - self.eps_clip, 1.0 + self.eps_clip) * advantages
            policy_actions, _ = self.actor.sample(states)
            actor_loss = -torch.min(surr1, surr2).mean() + lambda_risk * self.risk_model(states, policy_actions).mean()
            self.actor_optimizer.zero_grad(set_to_none=True)
            actor_loss.backward()
            self.actor_optimizer.step()
            for parameter in self.risk_model.parameters():
                parameter.requires_grad_(True)
            critic_loss = F.mse_loss(self.critic(states), returns)
            self.critic_optimizer.zero_grad(set_to_none=True)
            critic_loss.backward()
            self.critic_optimizer.step()
        self.memory.clear()

    def _prepare_training_data(self):
        with torch.no_grad():
            values = []
            for transition in self.memory:
                state_t = torch.as_tensor(transition[0], dtype=torch.float32, device=DEVICE).unsqueeze(0)
                values.append(float(self.critic(state_t).item()))
        values.append(0.0)
        gae = 0.0
        advantages = []
        returns = []
        for index in reversed(range(len(self.memory))):
            _, _, _, reward, _, done, _, _ = self.memory[index]
            delta = float(reward) + self.gamma * values[index + 1] * (1.0 - float(done)) - values[index]
            gae = delta + self.gamma * self.lam * (1.0 - float(done)) * gae
            advantages.insert(0, gae)
            returns.insert(0, gae + values[index])
        return [
            (transition[0], transition[1], transition[2], returns[index], advantages[index], transition[7])
            for index, transition in enumerate(self.memory)
        ]


def resolve_risk_cost(next_state, info):
    collision_indicator = extract_collision_indicator(info)
    if collision_indicator is not None:
        return calculate_risk_cost(next_state, collision_indicator)
    if "risk_cost" in info:
        return float(info["risk_cost"])
    raise KeyError("CARLA info must provide a collision indicator or a paper-aligned 'risk_cost'.")


def train_mixed_expert(env: CarlaEnv, max_episodes=50, max_steps=200, switch_interval=20, batch_size=64):
    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.shape[0]
    agent_ddpg = DDPGAgent(state_dim, action_dim)
    agent_sac = SACAgent(state_dim, action_dim)
    agent_ppo = PPOAgent(state_dim, action_dim)
    lambda_risk = {"DDPG": 0.8, "SAC": 0.4, "PPO": 0.1}
    for _ in range(max_episodes):
        state = env.reset()
        detections = env.detector.detect_objects(env)
        current_expert_name = emotion_to_expert(calculate_emotion_score(detections))
        done = False
        step_count = 0
        while not done and step_count < max_steps:
            step_count += 1
            if step_count % switch_interval == 0:
                detections = env.detector.detect_objects(env)
                current_expert_name = emotion_to_expert(calculate_emotion_score(detections))
            if current_expert_name == "DDPG":
                action = agent_ddpg.select_action(state)
                log_prob = None
            elif current_expert_name == "SAC":
                action = agent_sac.select_action(state)
                log_prob = None
            else:
                action, log_prob = agent_ppo.select_action(state)
            next_state, reward, done, info = env.step(action, current_expert_name)
            cost = resolve_risk_cost(next_state, info)
            if current_expert_name == "DDPG":
                agent_ddpg.store(state, action, reward, next_state, done, cost)
                agent_ddpg.update(batch_size, lambda_risk["DDPG"])
            elif current_expert_name == "SAC":
                agent_sac.store(state, action, reward, next_state, done, cost)
                agent_sac.update(batch_size, lambda_risk["SAC"])
            else:
                with torch.no_grad():
                    state_t = torch.as_tensor(state, dtype=torch.float32, device=DEVICE).unsqueeze(0)
                    value = float(agent_ppo.critic(state_t).item())
                agent_ppo.store((state, action, log_prob, reward, next_state, done, value, cost))
            state = next_state
        if len(agent_ppo.memory) >= 10:
            agent_ppo.update(lambda_risk["PPO"])
    return agent_ddpg, agent_sac, agent_ppo


def save_models(agent_ddpg, agent_sac, agent_ppo, model_dir):
    os.makedirs(model_dir, exist_ok=True)
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    torch.save(agent_ddpg.actor.state_dict(), os.path.join(model_dir, f"ddpg_actor_{timestamp}.pth"))
    torch.save(agent_ddpg.critic.state_dict(), os.path.join(model_dir, f"ddpg_critic_{timestamp}.pth"))
    torch.save(agent_ddpg.risk_model.state_dict(), os.path.join(model_dir, f"ddpg_risk_model_{timestamp}.pth"))
    torch.save(agent_sac.actor.state_dict(), os.path.join(model_dir, f"sac_actor_{timestamp}.pth"))
    torch.save(agent_sac.critic1.state_dict(), os.path.join(model_dir, f"sac_critic1_{timestamp}.pth"))
    torch.save(agent_sac.critic2.state_dict(), os.path.join(model_dir, f"sac_critic2_{timestamp}.pth"))
    torch.save(agent_sac.risk_model.state_dict(), os.path.join(model_dir, f"sac_risk_model_{timestamp}.pth"))
    torch.save(agent_ppo.actor.state_dict(), os.path.join(model_dir, f"ppo_actor_{timestamp}.pth"))
    torch.save(agent_ppo.critic.state_dict(), os.path.join(model_dir, f"ppo_critic_{timestamp}.pth"))
    torch.save(agent_ppo.risk_model.state_dict(), os.path.join(model_dir, f"ppo_risk_model_{timestamp}.pth"))


if __name__ == "__main__":
    detector = DetectorWithDepth()
    env = CarlaEnv(detector)
    try:
        calibrate_dpt_metric(env, detector)
        ddpg, sac, ppo = train_mixed_expert(
            env,
            max_episodes=2000,
            max_steps=600,
            switch_interval=20,
            batch_size=32,
        )
        save_models(ddpg, sac, ppo, "/root/autodl-fs/Uni yolo/save_models")
    finally:
        env.close()
