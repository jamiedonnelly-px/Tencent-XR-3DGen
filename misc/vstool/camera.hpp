//
// Created by Chen Tian on 2022/8/8.
//

#pragma

#include <opencv2/opencv.hpp>
class Camera {
public:
    Camera(bool is_tracking){
        intrinsic = cv::Mat::eye(3,3,CV_64FC1);
        m_is_tracking = is_tracking;
        if(!is_tracking)
        {
            distortion = cv::Mat::zeros(8, 1, CV_64FC1);
        }
        else
        {
		    distortion = cv::Mat::zeros(4, 1, CV_64FC1);
        }
        rotation = cv::Mat::zeros(3,3, CV_64FC1);
        translation = cv::Mat::zeros(3,1, CV_64FC1);
        transform = cv::Mat::eye(4,4,CV_64FC1);
    }
    ~Camera(){}
    const char* cam_name;
    int ID;
    cv::Size camera_size;
    cv::Mat intrinsic;
    cv::Mat distortion;
    cv::Mat rotation;
    cv::Mat translation;
    cv::Mat transform;
    bool m_is_tracking;

    void set_size(const char* size_string);
    void set_intrinsic(const char* focal_length_string, const char* principal_point_string);
    void set_radial_distortion(const char* radial_distortion_string);
    void set_transform(const char* translation_string, const char* rotation_string);
    std::vector<std::string> split(std::string str, std::string pattern);
};