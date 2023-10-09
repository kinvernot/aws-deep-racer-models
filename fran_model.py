# Imports
import math

def reward_function(params):

    # Repository: https://github.com/kinvernot/aws-deep-racer-models

    all_wheels_on_track = params['all_wheels_on_track']
    distance_from_center = params['distance_from_center']
    track_width = params['track_width']
    abs_steering_angle = abs(params['steering_angle'])  # Only need the absolute steering angle
    speed = params['speed']
    waypoints = params['waypoints']
    closest_waypoints = params['closest_waypoints']
    heading = params['heading']
    reward = math.exp(-6 * distance_from_center)  # negative exponential penalty
    # constants
    MAX_REWARD = 1e2
    MIN_REWARD = 1e-3
    DIRECTION_THRESHOLD = 10.0
    ABS_STEERING_THRESHOLD = 30

    def all_wheels_on_track_reward(current_reward):
        if not all_wheels_on_track:
            current_reward = MIN_REWARD
        else:
            current_reward = MAX_REWARD
        return current_reward

    def distance_from_center_reward_normalized(current_reward):
        # Normalize the car distance from center so we can use it in different tracks
        # As distance from center often stay around 0(center) to 0.5 of track_width.
        # any normDistance > 0.5 would indicate it is almost offtrack
        normDistance = distance_from_center / track_width
        return distance_from_center_reward(current_reward, normDistance)

    def distance_from_center_reward(current_reward, distance):
        # Calculate markers that are at varying distances away from the center line
        marker_1 = 0.2 * track_width
        marker_2 = 0.4 * track_width
        marker_3 = 0.7 * track_width
        marker_4 = 0.9 * track_width
        # Give higher reward if the car is closer to center line and vice versa
        if distance <= marker_1:
            current_reward *= 1.5
        elif distance <= marker_2:
            current_reward *= 1.1
        elif distance <= marker_3:
            current_reward *= 0.7
        elif distance <= marker_4:
            current_reward *= 0.3
        else:
            current_reward = MIN_REWARD  # likely crashed/ close to off track
        return current_reward

    def follow_the_center_line(current_reward):
        # Positive reward if the car is in a straight line going fast
        if abs_steering_angle < 0.1 and speed > 3:
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
        # Penalize if the difference is too large
        if direction_diff > DIRECTION_THRESHOLD:
            current_reward *= 0.5
        return current_reward

    def stay_inside_the_two_borders(current_reward):
        # Give a high reward if no wheels go off the track and the agent is somewhere in between the track borders
        if all_wheels_on_track and (0.5 * track_width - distance_from_center) >= 0.05:
            current_reward *= 1.2
        return current_reward

    def prevent_zig_zag(current_reward):
        # Penalize reward if the car is steering too much (your action space will matter)
        if abs_steering_angle > ABS_STEERING_THRESHOLD:
            current_reward += 0.8
        return current_reward

    def throttle(current_reward):
        # Decrease throttle while steering
        if speed > 2.5 - (0.4 * abs_steering_angle):
            current_reward *= 0.8
        return current_reward

    reward = all_wheels_on_track_reward(reward)
    reward = distance_from_center_reward_normalized(reward)
    # reward = distance_from_center_reward(reward, distance_from_center)
    reward = follow_the_center_line(reward)
    reward = direction(reward)
    reward = stay_inside_the_two_borders(reward)
    reward = prevent_zig_zag(reward)
    reward = throttle(reward)

    return float(reward)