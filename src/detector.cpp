
//
// Created by yamabuki on 2022/4/18.
//

#include <rm_detector/detector.h>

namespace rm_detector
{
Detector::Detector()
{
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

  roi_datas_pub_ = nh_.advertise<rm_msgs::RadarTargetDetection>("rm_radar/roi_datas", 10);
}

void Detector::receiveFromCam(const sensor_msgs::CompressedImageConstPtr& image)
{
  //    auto start = std::chrono::system_clock::now();
  cv_image_ = cv_bridge::toCvCopy(image, sensor_msgs::image_encodings::BGR8);

  car_inferencer_.detect(cv_image_->image);
  if (!car_inferencer_.target_objects.empty())
  {
    for (auto& object : car_inferencer_.target_objects)
    {
      cv::Mat armor_cls_image = cv_image_->image(get_rect(cv_image_->image, object.bbox)).clone();

      armor_inferencer_.detect(armor_cls_image);
      if (armor_inferencer_.target_objects.empty())
        continue;
      object.class_id = armor_inferencer_.target_objects[0].class_id;

      select_objects_.push_back(object);
    }
    if (turn_on_image_)
    {
      draw_bbox(cv_image_->image, select_objects_);
      camera_pub_.publish(cv_bridge::CvImage(std_msgs::Header(), "bgr8", cv_image_->image).toImageMsg());
    }
    if (!select_objects_.empty())
      publicMsg();
    else  // if no select_objects, it'll send empty msg to lidar
    {
      roi_array_.header.stamp = ros::Time::now();  // TODO: Whether the timestamp in camera info should be sent hear
      roi_array_.detections = {};
      roi_datas_pub_.publish(roi_array_);
    }
    select_objects_.clear();
    //        auto end = std::chrono::system_clock::now();
    //        ROS_INFO("inference time: %ld ms",std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count());
  }
}

void Detector::dynamicCallback(rm_detector::dynamicConfig& config)
{
  car_inferencer_.conf_thresh_ = config.g_conf_thresh;
  car_inferencer_.nms_thresh_ = config.g_nms_thresh;
  armor_inferencer_.conf_thresh_ = config.g_conf_thresh2;
  armor_inferencer_.nms_thresh_ = config.g_nms_thresh2;
  turn_on_image_ = config.g_turn_on_image;
  target_is_blue_ = config.target_is_blue;
  ROS_INFO("Settings have been seted");
}

void Detector::initalizeInfer()
{
  cudaSetDevice(kGpuId);
  car_inferencer_.init(car_model_path_, gLogger_);
  armor_inferencer_.init(armor_model_path_, gLogger_);
}

Detector::~Detector()
{
  this->roi_array_.detections.clear();
}

void Detector::publicMsg()
{  // enemy is blue
  std::vector<int> target;
  if (target_is_blue_)
  {
    target = { 0, 1, 2, 3, 4, 5, 11 };
  }
  else
  {  // red
    target = { 6, 7, 8, 9, 10, 11, 5 };
  }

  int car_size = select_objects_.size();

  rm_msgs::RadarTargetDetection roi_data;
  for (size_t i = 0; i < car_size; i++)
  {
    std::vector<int>::iterator iter;
    iter = std::find(target.begin(), target.end(), select_objects_[i].class_id);
    if (iter == target.end())
    {
      continue;
    }
    auto index = std::distance(target.begin(), iter);
    roi_data.id = index;

    roi_point_vec_.clear();

    float* box = select_objects_[i].bbox;

    roi_data_point_l_.x = box[0] - box[2] / 2;
    roi_data_point_l_.y = box[1] - box[3] / 2;
    roi_data_point_r_.x = box[0] + box[2] / 2;
    roi_data_point_r_.y = box[1] + box[3] / 2;

    roi_point_vec_.push_back(roi_data_point_l_);
    roi_point_vec_.push_back(roi_data_point_r_);

    roi_data.position.data.push_back(roi_point_vec_[0].x);
    roi_data.position.data.push_back(roi_point_vec_[0].y);
    roi_data.position.data.push_back(roi_point_vec_[1].x);
    roi_data.position.data.push_back(roi_point_vec_[1].y);
    roi_array_.detections.emplace_back(roi_data);
  }
  roi_array_.header.stamp = ros::Time::now();  // TODO: Whether the timestamp in camera info should be sent hear
  roi_datas_pub_.publish(roi_array_);
}
}  // namespace rm_detector
PLUGINLIB_EXPORT_CLASS(rm_detector::Detector, nodelet::Nodelet)