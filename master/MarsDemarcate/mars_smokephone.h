#pragma once
#include <fstream>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include "include/net.h"
#include <iostream>

#include <stdio.h>
using namespace std;
//ofstream messagefile;
void init_nose_arm_mark(const char* deploy,const char* weight);
bool key_people_num(cv::Mat image_gray,int & act_peonum_temp,const char* deploy,const char* weight);

bool key_point(cv::Mat image,cv::Point2f& nose_p_from_keypoint_temp,float& rear_angle,float& turn_face, float& turn_ear, int & act_peonum_temp);
