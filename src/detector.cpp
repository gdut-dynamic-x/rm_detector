
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

    nh_.getParam("roi_data1_name", roi_data1_name_);
    nh_.getParam("roi_data2_name", roi_data2_name_);
    nh_.getParam("roi_data3_name", roi_data3_name_);
    nh_.getParam("roi_data4_name", roi_data4_name_);
    nh_.getParam("roi_data5_name", roi_data5_name_);
    nh_.getParam("roi_data6_name", roi_data6_name_);
    nh_.getParam("roi_data7_name", roi_data7_name_);
    nh_.getParam("left_camera", left_camera_);

    initalizeInfer();

    ros::NodeHandle nh_reconfig(nh_, nodelet_name_ + "_reconfig");
    server_ = new dynamic_reconfigure::Server<rm_detector::dynamicConfig>(nh_reconfig);
    callback_ = boost::bind(&Detector::dynamicCallback, this, _1);
    server_->setCallback(callback_);

    if (left_camera_)
        camera_sub_ = nh_.subscribe("/galaxy_camera/galaxy_camera_left/image_raw/compressed", 1, &Detector::receiveFromCam, this);
    else
        camera_sub_ = nh_.subscribe("/galaxy_camera/galaxy_camera_right/image_raw/compressed", 1, &Detector::receiveFromCam, this);

    camera_pub_ = nh_.advertise<sensor_msgs::Image>(camera_pub_name_, 1);

    roi_data_pub1_ = nh_.advertise<std_msgs::Float32MultiArray>(roi_data1_name_, 1);
    roi_data_pub2_ = nh_.advertise<std_msgs::Float32MultiArray>(roi_data2_name_, 1);
    roi_data_pub3_ = nh_.advertise<std_msgs::Float32MultiArray>(roi_data3_name_, 1);
    roi_data_pub4_ = nh_.advertise<std_msgs::Float32MultiArray>(roi_data4_name_, 1);
    roi_data_pub5_ = nh_.advertise<std_msgs::Float32MultiArray>(roi_data5_name_, 1);
    roi_data_pub6_ = nh_.advertise<std_msgs::Float32MultiArray>(roi_data6_name_, 1);
    roi_data_pub7_ = nh_.advertise<std_msgs::Float32MultiArray>(roi_data7_name_, 1);

    roi_data_pub_vec.push_back(roi_data_pub1_);
    roi_data_pub_vec.push_back(roi_data_pub2_);
    roi_data_pub_vec.push_back(roi_data_pub3_);
    roi_data_pub_vec.push_back(roi_data_pub4_);
    roi_data_pub_vec.push_back(roi_data_pub5_);
    roi_data_pub_vec.push_back(roi_data_pub6_);
    roi_data_pub_vec.push_back(roi_data_pub7_);

}

void Detector::receiveFromCam(const sensor_msgs::CompressedImageConstPtr &image) {
//    auto start = std::chrono::system_clock::now();
    cv_bridge::CvImagePtr cv_image_ = cv_bridge::toCvCopy(image, sensor_msgs::image_encodings::BGR8);

    carInferencer.detect(cv_image_->image);
    if (!carInferencer.target_objects.empty()) {
        for (auto &object: carInferencer.target_objects) {
            cv::Mat armor_cls_image = cv_image_->image(get_rect(cv_image_->image, object.bbox)).clone();

            armorInferencer.detect(armor_cls_image);
            if (armorInferencer.target_objects.empty()) continue;
            object.class_id = armorInferencer.target_objects[0].class_id;

            select_objects.push_back(object);

        }
        if (turn_on_image_) {
            draw_bbox(cv_image_->image, select_objects);
            camera_pub_.publish(cv_bridge::CvImage(std_msgs::Header(), "bgr8", cv_image_->image).toImageMsg());
        }
        if (!select_objects.empty())
            publicMsg();
        select_objects.clear();
//        auto end = std::chrono::system_clock::now();
//        ROS_INFO("inference time: %ld ms",std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count());
    }
}

void Detector::dynamicCallback(rm_detector::dynamicConfig& config)
{
    carInferencer.conf_thresh_ = config.g_conf_thresh;
    carInferencer.nms_thresh_ = config.g_nms_thresh;
    armorInferencer.conf_thresh_ = config.g_conf_thresh2;
    armorInferencer.nms_thresh_ = config.g_nms_thresh2;
    turn_on_image_ = config.g_turn_on_image;
    target_is_blue_ = config.target_is_blue;
    ROS_INFO("Settings have been seted");
}

void Detector::initalizeInfer()
{
    cudaSetDevice(kGpuId);
    carInferencer.init(car_model_path_, gLogger_);
    armorInferencer.init(armor_model_path_, gLogger_);
}

Detector::~Detector()
{
}

void Detector::publicMsg() {//enemy is blue
    std::vector<int> target;
    if (target_is_blue_){
        target = {0, 1, 2, 3, 4, 5, 11};
    } else {//red
        target = {6, 7, 8, 9, 10, 11, 5};
    }

    int car_size = select_objects.size();

    for (size_t i = 0; i < car_size; i++)
    {
        std::vector<int>::iterator iter;
        iter = std::find(target.begin(), target.end(), select_objects[i].class_id);
        if (iter == target.end()) {
            continue;
        }
        auto index = std::distance(target.begin(), iter);

        roi_point_vec_.clear();
        roi_data_.data.clear();

        float* box = select_objects[i].bbox;

        roi_data_point_l_.x = box[0] - box[2] / 2;
        roi_data_point_l_.y = box[1] - box[3] / 2;
        roi_data_point_r_.x = box[0] + box[2] / 2;
        roi_data_point_r_.y = box[1] + box[3] / 2;

        roi_point_vec_.push_back(roi_data_point_l_);
        roi_point_vec_.push_back(roi_data_point_r_);

        roi_data_.data.push_back(roi_point_vec_[0].x);
        roi_data_.data.push_back(roi_point_vec_[0].y);
        roi_data_.data.push_back(roi_point_vec_[1].x);
        roi_data_.data.push_back(roi_point_vec_[1].y);

        roi_data_pub_vec[index].publish(roi_data_);
    }
}
}  // namespace rm_detector
PLUGINLIB_EXPORT_CLASS(rm_detector::Detector, nodelet::Nodelet)