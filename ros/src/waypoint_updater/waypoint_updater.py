#!/usr/bin/env python

import rospy
from geometry_msgs.msg import PoseStamped
from styx_msgs.msg import Lane, Waypoint
from scipy.spatial import KDTree

from std_msgs.msg import Int32

import numpy as np
import math

'''
This node will publish waypoints from the car's current position to some `x` distance ahead.

As mentioned in the doc, you should ideally first implement a version which does not care
about traffic lights or obstacles.

Once you have created dbw_node, you will update this node to use the status of traffic lights too.

Please note that our simulator also provides the exact location of traffic lights and their
current status in `/vehicle/traffic_lights` message. You can use this message to build this node
as well as to verify your TL classifier.

TODO (for Yousuf and Aaron): Stopline location for each traffic light.
'''

MAX_DECEL = 0.5
LOOKAHEAD_WPS = 50 # Number of waypoints we will publish. You can change this number
PUBLISHING_RATE = 50 # [Hz]
STOP_LINE_BUFFER = 3

class WaypointUpdater(object):
    def __init__(self):
        rospy.init_node('waypoint_updater')
        
        # TODO: Add other member variables you need below
        self.base_lane       = None
        self.pose            = None
        self.stopline_wp_idx = -1
        self.waypoints_2d    = None
        self.waypoint_tree   = None
        
        # Subscriber to get the waypoints
        rospy.Subscriber('/current_pose', PoseStamped, self.pose_cb)
        
        # Waypoint callback to get base_waypoints as `Lane` msg
        rospy.Subscriber('/base_waypoints', Lane, self.waypoints_cb)
        # rospy subscriber format: rospy.Subscriber(<topic>, <MsgType>, <callback>)
        
        # New subscriber
        rospy.Subscriber('/traffic_waypoint', Int32, self.traffic_cb)

        # TODO: Add a subscriber for /traffic_waypoint and /obstacle_waypoint below
        self.final_waypoints_pub = rospy.Publisher('final_waypoints', Lane, queue_size=1)

        # Replace the rospy.spin() with a custom loop() function
        # to control of publishing frequency
        self.loop()

    def loop(self):
        # Set the target publish rate to 50Hz
        rate = rospy.Rate(PUBLISHING_RATE)
        while not rospy.is_shutdown():
            if self.pose and self.base_lane:
                self.publish_waypoints()
            rate.sleep()

    def get_closest_waypoint_idx(self):
        x = self.pose.pose.position.x
        y = self.pose.pose.position.y
        
        closest_idx = self.waypoint_tree.query([x,y],1)[1]

        # Check if closest is ahead or behind vehicle
        closest_coord = self.waypoints_2d[closest_idx]
        prev_coord    = self.waypoints_2d[closest_idx-1]

        # Equations for hyperplane through clsoest_coords
        cl_vect   = np.array(closest_coord)
        prev_vect = np.array(prev_coord)
        pos_vect  = np.array([x, y])

        '''
        Find the next clsoest waypoints if it's behind the car
            val > 0: waypoint ahead of car
            val < 0: waypoint behind
        '''
        val = np.dot(cl_vect-prev_vect, pos_vect-cl_vect)

        # Get the next closest waypoint +1 if the current waypoint
        # is behind us (i.e. val > 0)
        if val > 0:
            closest_idx = (closest_idx + 1) % len(self.waypoints_2d)
        return closest_idx
   
    def decelerate_waypoints(self, waypoints, closest_idx):
        temp = []
        for i,wp in enumerate(waypoints):
            '''
            Enumerate over sliced list of waypoints.
            Create a new waypoint message, set the pose to the base waypoint pose.
            Position of waypoint doesn't change orientation.     
            '''
            p = Waypoint()
            p.pose = wp.pose
            
            '''
            Find the exact center of the car
            Note if the car does not have -2 from the stop line,
            the car will stop with the center of the car right on the line
            
            It is a good idea to put -2 or -3 to make the nose of the car stop
            behind the line
            '''
            # Two waypoints back from line so front of car stops at line
            stop_idx = max(self.stopline_wp_idx - closest_idx - STOP_LINE_BUFFER, 0)

            '''
            Slice the waypoints from the closest index to number of LOOKAHEAD_WPS.
            Calculate how far from this light that we need to stop at
            by using distance function that does linear piecewise distance and summing
            up all the line segments between waypoints.
            
            distance = 0 if i > stop_idx
            
            '''
            dist = self.distance(waypoints, i, stop_idx)
            vel = math.sqrt(2 * MAX_DECEL * dist)
            '''
            Set waypoint velocity to zero 
            when waypoint is reached
            
            Similar to the waypoint loader code that waypoints are decelerated at the beginning,
            it accelerates from zero to regular waypoint velocity, and then at the end of a waypoint list,
            it's decelerating to zeroa again.
            
            As we get close to the stop index, the distance becomes very small and velocity goes to zero.
            
            Square root is steep when it get close to the stop line.
            
            Instead of sqrt, one can replace it multiply by a constant to get some linearty.
            '''
            
            # Set velocity to zero if it's small enough
            if vel < 1.0:
                vel = 0.0
            
            '''
            sqroot can become very large as the distance is large. Don't want to set a large velocity if we're a long way away from stop waypoint. Rather keep the velocity that was given for the waypoints before. 
            '''
            # Keep the speed limit, limit all velocities on the waypoints
            # as the square root becomes smaller, switch over to square root velocity
            p.twist.twist.linear.x = min(vel, wp.twist.twist.linear.x)
            
            # Append and return the list of newly created waypoints for our lane
            temp.append(p)
            
        return temp
    
    def generate_lane(self):
        '''
        Generate lane function take the waypoints and 
        update their velocities based on how we want the car to behave.
        
        If we have some traffic lights coming in, slow down the car 
        leading up to the stop line in front of the traffic light.
        
        Changing the waypoint's property by setting the velocity value (twist.linear.x)
        with the target velocity and some extra logic to take the car
        to the target speed.
        
        '''
        lane = Lane()
        
        closest_idx    = self.get_closest_waypoint_idx()
        farthest_idx   = closest_idx + LOOKAHEAD_WPS
        base_waypoints = self.base_lane.waypoints[closest_idx:farthest_idx]
        
        '''
        #################################################################
        # !!! ATN !!! BE CAREUL WHEN WORKING WITH LARGE WAYPOINT LIST   #
        #################################################################
        
        NOTE: If you have a very large list, make sure slice the base lane waypoints before you start updating them. 
        So if you iterate through the entire set of baseline waypoints each time and do this decelerate_waypoints() update,
        it would not be efficent. And it can introduce a lot of latency in the code. 
        And then by the time the updated waypoints actually get to the car, the car's moved past them.
        
        Be efficient with speed, so make sure to slice before hand to avoid any latency. 
        '''
        
        '''
        If the stop line waypoint is not found or it's further than the
        farthest index, then publish the base waypoints because no traffic light data
        detected we're concerned about
        '''
        if self.stopline_wp_idx == -1 or (self.stopline_wp_idx >= farthest_idx):
            lane.waypoints = base_waypoints
        else:
            '''
            Else there is a valid stop line waypoint and its within the farthest waypoint
            index, then get the base waypoints to the closest index.
            
            Note the declerate waypoints has a different message types than the
            regular message types.
            The reason creating a new waypoint message type because we don't want to modify
            the base waypoints because that message comes in only once and we want to keep the
            base waypoints preserved. 
            
            If we modify the base waypoints, then we lose them if you drive back over the same waypoints.
            So we want to create new list of waypoints but use some of the information from those base waypoints.
            
            '''
            lane.waypoints = self.decelerate_waypoints(base_waypoints, closest_idx)
            
        return lane
        
    def publish_waypoints(self):
        '''
        Publish final lane directly
        '''
        final_lane = self.generate_lane()
        self.final_waypoints_pub.publish(final_lane)

    def pose_cb(self, msg):
        # TODO: Implement
        # Store the car's pose
        self.pose = msg

    # Waypoints callback
    def waypoints_cb(self, waypoints):
        # TODO: Implement
        # Set the base waypoints once because it never change after that
        self.base_lane = waypoints

        # Take first LOOKAHEAD_WPS (i.e. 200) points of way points ahead of car
        # Prevent race condition: make sure the self.waypoints_2d is initalized before subscriber
        if not self.waypoints_2d:
            self.waypoints_2d = [[waypoint.pose.pose.position.x, waypoint.pose.pose.position.y] for waypoint in waypoints.waypoints]
            self.waypoint_tree = KDTree(self.waypoints_2d)

    def traffic_cb(self, msg):
        # TODO: Callback for /traffic_waypoint message. Implement
        #Store the stop line waypoint index
        self.stopline_wp_idx = msg.data

    def obstacle_cb(self, msg):
        # TODO: Callback for /obstacle_waypoint message. We will implement it later
        pass

    def get_waypoint_velocity(self, waypoint):
        return waypoint.twist.twist.linear.x

    def set_waypoint_velocity(self, waypoints, waypoint, velocity):
        waypoints[waypoint].twist.twist.linear.x = velocity

    def distance(self, waypoints, wp1, wp2):
        dist = 0
        dl = lambda a, b: math.sqrt((a.x-b.x)**2 + (a.y-b.y)**2  + (a.z-b.z)**2)
        for i in range(wp1, wp2+1):
            dist += dl(waypoints[wp1].pose.pose.position, waypoints[i].pose.pose.position)
            wp1 = i
        return dist


if __name__ == '__main__':
    try:
        WaypointUpdater()
    except rospy.ROSInterruptException:
        rospy.logerr('Could not start waypoint updater node.')
