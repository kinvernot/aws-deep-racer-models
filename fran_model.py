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
    steps = params['steps']
    progress = params['progress']

    # constants
    MIN_REWARD = 1e-3
    # depends on training configuration
    ABS_STEERING_THRESHOLD = 30
    SPEED_THRESHOLD = 2.5

    def all_wheels_on_track_and_steps_reward(current_reward):
        if all_wheels_on_track and steps > 0:
            # motivated the model to stay on the track and get around in as few steps as possible
            current_reward = ((progress / steps) * 100) + speed ** 2
        else:
            current_reward = MIN_REWARD
        return current_reward

    def distance_from_center_normalized_reward(current_reward):
        # Normalize the car distance from center so we can use it in different tracks
        # As distance from center often stay around 0(center) to 0.5 of track_width.
        # any normalized_distance > 0.5 would indicate it is almost offtrack
        normalized_distance = distance_from_center / track_width
        return distance_from_center_reward(current_reward, normalized_distance)

    def distance_from_center_reward(current_reward, distance):
        # Give higher reward if the car is closer to center line and vice versa
        if distance <= track_width * 0.1:
            current_reward *= 1.5
        elif distance <= track_width * 0.2:
            current_reward *= 1.2
        elif distance <= track_width * 0.3:
            current_reward *= 1.0
        elif distance <= track_width * 0.5:
            current_reward *= 0.9
        else:
            # likely crashed/ close to off track [any normalized_distance > 0.5 would indicate it is almost offtrack]
            current_reward = MIN_REWARD
        return current_reward

    def straight_line_going_fast(current_reward):
        # Positive reward if the car is in a straight line going fast
        if abs_steering_angle < 0.1 and speed > (SPEED_THRESHOLD * 0.8):
            current_reward *= 1.3
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
        direction_difference = abs(suggested_direction - heading)

        if direction_difference <= 0.1:
            current_reward *= 1.6
        if direction_difference <= 1.0:
            current_reward *= 1.3
        elif direction_difference <= 2.0:
            current_reward *= 1.2
        elif direction_difference <= 3.0:
            current_reward *= 1.1
        elif direction_difference <= 4.0:
            current_reward *= 1.0
        elif direction_difference <= 5.0:
            current_reward *= 0.9
        elif direction_difference <= 10.0:
            current_reward *= 0.8
        else:
            current_reward *= 0.5

        return current_reward

    def prevent_zig_zag(current_reward):
        # Penalize reward if the car is steering too much (your action space will matter)
        if abs_steering_angle > ABS_STEERING_THRESHOLD:
            current_reward *= 0.8
        # Decrease throttle while steering
        if speed > (SPEED_THRESHOLD * 0.6) - (0.4 * abs_steering_angle):
            current_reward *= 0.8
        return current_reward

    reward = 0.0
    reward = all_wheels_on_track_and_steps_reward(reward)
    reward = direction(reward)
    reward = distance_from_center_normalized_reward(reward)
    reward = prevent_zig_zag(reward)
    reward = straight_line_going_fast(reward)

    return float(reward)