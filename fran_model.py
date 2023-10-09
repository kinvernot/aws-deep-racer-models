# Imports
import math


def reward_function(params):

    # Repository: https://github.com/kinvernot/aws-deep-racer-models

    # Read input parameters
    all_wheels_on_track = params['all_wheels_on_track']
    distance_from_center = params['distance_from_center']
    track_width = params['track_width']
    abs_steering_angle = abs(params['steering_angle'])  # Only need the absolute steering angle
    speed = params['speed']
    waypoints = params['waypoints']
    closest_waypoints = params['closest_waypoints']
    heading = params['heading']
    reward = math.exp(-6 * distance_from_center)  # negative exponential penalty
    steps = params['steps']
    progress = params['progress']

    # constants
    MAX_REWARD = 1e2
    MIN_REWARD = 1e-3
    ABS_STEERING_THRESHOLD = 30
    STEPS_THRESHOLD = 300
    SPEED_THRESHOLD = 2.0

    def all_wheels_on_track_reward(current_reward):
        if not all_wheels_on_track:
            current_reward = MIN_REWARD
        else:
            current_reward = MAX_REWARD
        return current_reward

    def distance_from_center_normalized_reward(current_reward):
        # Normalize the car distance from center so we can use it in different tracks
        # As distance from center often stay around 0(center) to 0.5 of track_width.
        # any normalized_distance > 0.5 would indicate it is almost offtrack
        normalized_distance = distance_from_center / track_width
        return distance_from_center_reward(current_reward, normalized_distance)

    def distance_from_center_reward(current_reward, distance):
        # Calculate markers that are at varying distances away from the center line
        marker_1 = 0.1 * track_width
        marker_2 = 0.2 * track_width
        marker_3 = 0.3 * track_width
        marker_4 = 0.5 * track_width
        # Give higher reward if the car is closer to center line and vice versa
        if distance <= marker_1:
            current_reward *= 1.5
        elif distance <= marker_2:
            current_reward *= 1.2
        elif distance <= marker_3:
            current_reward *= 0.9
        elif distance <= marker_4:
            current_reward *= 0.7
        else:
            current_reward = MIN_REWARD  # likely crashed/ close to off track
        return current_reward

    def straight_line_going_fast(current_reward):
        # Positive reward if the car is in a straight line going fast
        if abs_steering_angle < 0.1 and speed > (SPEED_THRESHOLD * 0.8):
            current_reward *= 1.2
        return current_reward

    def direction(current_reward):
        # Calculate the direction of the center line based on the closest waypoints
        next_point = waypoints[int(closest_waypoints[1])]
        prev_point = waypoints[int(closest_waypoints[0])]
        # Calculate the direction in radius, arctan2(dy, dx), the result is (-pi, pi) in radians
        suggested_direction = math.atan2(next_point[1] - prev_point[1], next_point[0] - prev_point[0])
        # Convert to degrees
        suggested_direction = math.degrees(suggested_direction)
        # Calculate difference between track direction and car heading angle
        direction_diff = abs(suggested_direction - heading)

        marker_direction = 0.1
        marker_direction_1 = 1.0
        marker_direction_2 = 2.0
        marker_direction_3 = 3.0
        marker_direction_4 = 4.0
        marker_direction_5 = 5.0
        marker_direction_10 = 10.0

        if direction_diff <= marker_direction:
            current_reward *= 1.6
        if direction_diff <= marker_direction_1:
            current_reward *= 1.3
        elif direction_diff <= marker_direction_2:
            current_reward *= 1.2
        elif direction_diff <= marker_direction_3:
            current_reward *= 1.1
        elif direction_diff <= marker_direction_4:
            current_reward *= 1.0
        elif direction_diff <= marker_direction_5:
            current_reward *= 0.9
        elif direction_diff <= marker_direction_10:
            current_reward *= 0.8
        else:
            current_reward *= 0.5

        return current_reward

    def steps_reward(current_reward):
        # Give additional reward if the car pass every 50 steps faster than expected
        if (steps % 50) == 0 and progress >= (steps / STEPS_THRESHOLD) * 100:
            current_reward *= 1.2
        return current_reward

    def prevent_zig_zag(current_reward):
        # Penalize reward if the car is steering too much (your action space will matter)
        if abs_steering_angle > ABS_STEERING_THRESHOLD:
            current_reward *= 0.8
        return current_reward

    def throttle(current_reward):
        # Decrease throttle while steering
        if speed > (SPEED_THRESHOLD * 0.6) - (0.4 * abs_steering_angle):
            current_reward *= 0.8
        return current_reward

    reward = all_wheels_on_track_reward(reward)
    reward = distance_from_center_normalized_reward(reward)
    reward = straight_line_going_fast(reward)
    reward = direction(reward)
    reward = steps_reward(reward)
    reward = prevent_zig_zag(reward)
    reward = throttle(reward)

    return float(reward)