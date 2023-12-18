//
// Created by yamabuki on 2022/4/18.
//
#pragma once
#include <iostream>
#include <opencv2/opencv.hpp>
#include <ros/ros.h>
#include <cv_bridge/cv_bridge.h>
#include <string>
#include <vector>
#include <std_msgs/Float32MultiArray.h>
#include <dynamic_reconfigure/server.h>
#include "rm_detector/dynamicConfig.h"
#include <sensor_msgs/CameraInfo.h>
#include <nodelet/nodelet.h>
#include <pluginlib/class_list_macros.h>
#include "rm_detector/inferencer.h"


//static Logger gLogger_;

namespace rm_detector
{
class Detector : public nodelet::Nodelet
{
public:
    Detector();

    ~Detector() override;

    void onInit() override;

    void receiveFromCam(const sensor_msgs::CompressedImageConstPtr & image);

    void publicMsg();

    void initalizeInfer();

    void dynamicCallback(rm_detector::dynamicConfig& config);

    cv_bridge::CvImagePtr cv_image_;

    std_msgs::Float32MultiArray roi_data_;
    std::vector<cv::Point2f> roi_point_vec_;
    cv::Point2f roi_data_point_r_;
    cv::Point2f roi_data_point_l_;

    std::string car_model_path_;
    std::string armor_model_path_;


    bool turn_on_image_;
    dynamic_reconfigure::Server<rm_detector::dynamicConfig>* server_;
    dynamic_reconfigure::Server<rm_detector::dynamicConfig>::CallbackType callback_;

    std::string camera_pub_name_;
    std::string nodelet_name_;

    std::string roi_data1_name_;    //敌方英雄
    std::string roi_data2_name_;    //敌方工程
    std::string roi_data3_name_;    //敌方步兵
    std::string roi_data4_name_;    //敌方步兵
    std::string roi_data5_name_;    //敌方步兵
    std::string roi_data6_name_;    //敌方烧饼
    std::string roi_data7_name_;    //我方烧饼

    bool target_is_blue_;
    bool left_camera_;

    Inferencer carInferencer;
    Inferencer armorInferencer;

    std::vector<Detection> select_objects;

    ros::NodeHandle nh_;
    Logger gLogger_;


private:

    ros::Publisher camera_pub_;

    ros::Subscriber camera_sub_;

    std::vector<ros::Publisher> roi_data_pub_vec;
    ros::Publisher roi_data_pub1_;
    ros::Publisher roi_data_pub2_;
    ros::Publisher roi_data_pub3_;
    ros::Publisher roi_data_pub4_;
    ros::Publisher roi_data_pub5_;
    ros::Publisher roi_data_pub6_;
    ros::Publisher roi_data_pub7_;

};
}  // namespace rm_detector