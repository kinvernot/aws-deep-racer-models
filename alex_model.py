#this model behaves well with a action space of speed between 1-2ms
import math

def reward_function(params):
    on_track = params['all_wheels_on_track']
    distance_from_center = params['distance_from_center']
    track_width = params['track_width']
    steering = abs(params['steering_angle'])
    speed = params['speed']
    waypoints = params['waypoints']
    closest_waypoints = params['closest_waypoints']
    heading = params['heading']

    reward = math.exp(-6 * distance_from_center)

    def on_track_reward(current_reward, on_track):
        if not on_track:
            current_reward = 1e-3
        else:
            current_reward = 1e2
        return current_reward

    def distance_from_center_reward(current_reward, track_width, distance_from_center):
        marker_1 = 0.1 * track_width
        marker_2 = 0.25 * track_width
        marker_3 = 0.5 * track_width

        if distance_from_center <= marker_1:
            current_reward *= 1.2
        elif distance_from_center <= marker_2:
            current_reward *= 0.8
        elif distance_from_center <= marker_3:
            current_reward += 0.5
        else:
            current_reward = 1e-3

        return current_reward

    def straight_line_reward(current_reward, steering, speed):
        if abs(steering) < 0.1 and speed > 3:
            current_reward *= 1.2
        return current_reward

    def direction_reward(current_reward, waypoints, closest_waypoints, heading):
        next_point = waypoints[int(closest_waypoints[1])]
        prev_point = waypoints[int(closest_waypoints[0])]

        direction = math.atan2(next_point[1] - prev_point[1], next_point[0] - prev_point[0])
        direction = math.degrees(direction)

        direction_diff = abs(direction - heading)

        if direction_diff > 10.0:
            current_reward *= 0.5

        return current_reward
    
    def speed_reward(current_reward,speed):
        next_point = waypoints[int(closest_waypoints[1])]
        prev_point = waypoints[int(closest_waypoints[0])]

        direction = math.atan2(next_point[1] - prev_point[1], next_point[0] - prev_point[0])
        direction = math.degrees(direction)

        direction_diff = abs(direction - heading)

        if direction_diff <= 10.0:
            if speed > 3:
                current_reward *= 1.5
        else:
            if speed > 1.5:
                current_reward *= 0.5
            else:
                current_reward *= 0.3

        return current_reward

    reward = on_track_reward(reward, on_track)
    reward = distance_from_center_reward(reward, track_width, distance_from_center)
    reward = straight_line_reward(reward, steering, speed)
    reward = direction_reward(reward, waypoints, closest_waypoints, heading)
    reward = speed_reward(reward, speed)

    return float(reward)
