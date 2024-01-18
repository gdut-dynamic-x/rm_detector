
//
// Created by yamabuki on 2022/4/18.
//

#include <rm_detector/detector.h>
#include "rm_msgs/RadarTargetDetection.h"
#include "rm_msgs/RadarTargetDetectionArray.h"

namespace rm_detector
{
Detector::Detector()
{
  num_frame_ = 0;
  total_ms_ = 0;
  tracker_ = new rm_bytetrack::BYTETracker(50, 100);
}

void Detector::onInit()
{
  nh_ = getMTPrivateNodeHandle();
  nh_.getParam("g_car_model_path", car_model_path_);
  nh_.getParam("g_armor_model_path", armor_model_path_);

  nh_.getParam("camera_pub_name", camera_pub_name_);
  nh_.getParam("nodelet_name", nodelet_name_);

  nh_.getParam("left_camera", left_camera_);

  initalizeInfer();

  ros::NodeHandle nh_reconfig(nh_, nodelet_name_ + "_reconfig");
  server_ = new dynamic_reconfigure::Server<rm_detector::dynamicConfig>(nh_reconfig);
  callback_ = boost::bind(&Detector::dynamicCallback, this, _1);
  server_->setCallback(callback_);

  if (left_camera_)  // TODO: Should we use the subscribeCamera function to receive camera info?
    camera_sub_ =
        nh_.subscribe("/galaxy_camera/galaxy_camera_left/image_raw/compressed", 1, &Detector::receiveFromCam, this);
  else
    camera_sub_ =
        nh_.subscribe("/galaxy_camera/galaxy_camera_right/image_raw/compressed", 1, &Detector::receiveFromCam, this);

  camera_pub_ = nh_.advertise<sensor_msgs::Image>(camera_pub_name_, 1);

  camera_pub_track_ = nh_.advertise<sensor_msgs::Image>(camera_pub_name_ + "_track_", 1);

  roi_datas_pub_ = nh_.advertise<rm_msgs::RadarTargetDetectionArray>("rm_radar/roi_datas", 10);
}

void Detector::receiveFromCam(const sensor_msgs::CompressedImageConstPtr& image)
{
  if (num_frame_ > 1000)
  {
    num_frame_ = 0;
    total_ms_ = 0;
  }
  num_frame_++;
  auto start = std::chrono::system_clock::now();
  cv_image_ = cv_bridge::toCvCopy(image, sensor_msgs::image_encodings::BGR8);

  car_inferencer_.detect(cv_image_->image);

  if (!car_inferencer_.target_objects_.empty())
  {
    for (auto& object : car_inferencer_.target_objects_)
    {
      cv::Mat armor_cls_image = cv_image_->image(get_rect(cv_image_->image, object.bbox)).clone();

      armor_inferencer_.detect(armor_cls_image);
      if (armor_inferencer_.target_objects_.empty())
      {
        object.class_id = -1;
        continue;
      }
      object.class_id = armor_inferencer_.target_objects_[0].class_id;
      object.conf = 1.0f;
    }

    std::vector<Object> objects;
    for (auto& targetObject : car_inferencer_.target_objects_)
    {
      Object object;
      object.rect = get_rect(cv_image_->image, targetObject.bbox);
      object.label = targetObject.class_id;
      object.prob = targetObject.conf;
      objects.push_back(object);
    }
    output_stracks_.clear();
    output_stracks_ = tracker_->update(objects);

    if (turn_on_image_)
    {
      cv::Mat img_clone = cv_image_->image.clone();
      draw_bbox(img_clone, car_inferencer_.target_objects_);
      camera_pub_.publish(cv_bridge::CvImage(std_msgs::Header(), "bgr8", img_clone).toImageMsg());
      auto end = std::chrono::system_clock::now();
      total_ms_ = total_ms_ + std::chrono::duration_cast<std::chrono::microseconds>(end - start).count();
      for (int i = 0; i < output_stracks_.size(); i++)
      {
        std::vector<float> tlwh = output_stracks_[i].tlwh_;
        putText(cv_image_->image, cv::format("%d", output_stracks_[i].track_class_id_), cv::Point(tlwh[0], tlwh[1] - 5),
                0, 0.6, cv::Scalar(0, 0, 255), 2, cv::LINE_AA);
        rectangle(cv_image_->image, cv::Rect(tlwh[0], tlwh[1], tlwh[2], tlwh[3]), cv::Scalar(255, 0, 0), 2);
      }
      putText(cv_image_->image,
              cv::format("frame: %d fps: %d num: %d", num_frame_, num_frame_ * 1000000 / total_ms_,
                         output_stracks_.size()),
              cv::Point(0, 30), 0, 0.6, cv::Scalar(0, 0, 255), 2, cv::LINE_AA);
      camera_pub_track_.publish(cv_bridge::CvImage(std_msgs::Header(), "bgr8", cv_image_->image).toImageMsg());
    }

    if (!output_stracks_.empty())
      publicMsg();
  }
}

void Detector::dynamicCallback(rm_detector::dynamicConfig& config)
{
  car_inferencer_.conf_thresh_ = config.g_car_conf_thresh;
  car_inferencer_.nms_thresh_ = config.g_car_nms_thresh;
  armor_inferencer_.conf_thresh_ = config.g_armor_conf_thresh;
  armor_inferencer_.nms_thresh_ = config.g_armor_nms_thresh;
  turn_on_image_ = config.g_turn_on_image;
  target_is_blue_ = config.target_is_blue;
  ROS_INFO("Settings have been seted");
}

void Detector::initalizeInfer()
{
  cudaSetDevice(kGpuId);
  car_inferencer_.init(car_model_path_, gLogger_);
  armor_inferencer_.init(armor_model_path_, gLogger_);

  //  tracker_ = new rm_bytetrack::BYTETracker(50, 100);
}

Detector::~Detector()
{
  this->roi_array_.detections.clear();
}

void Detector::publicMsg()
{
  rm_msgs::RadarTargetDetectionArray array;
  array.header.stamp = ros::Time::now();
  for (auto& output_strack : output_stracks_)
  {
    rm_msgs::RadarTargetDetection data;
    data.id = output_strack.track_class_id_;
    data.position.data.assign(output_strack.tlbr_.begin(), output_strack.tlbr_.end());
    array.detections.push_back(data);
  }
  roi_datas_pub_.publish(array);
}
}  // namespace rm_detector
PLUGINLIB_EXPORT_CLASS(rm_detector::Detector, nodelet::Nodelet)