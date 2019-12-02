#pragma once

#include <opencv2/opencv.hpp>
#include "line.h"

#include "Dstmap.h"
struct adas_camera
{
	/*****camera***********/
	int _center_x = 828;
	int _center_y = 844;
	int _radius = 318;//Ô¤Áô
					  /****lane box***********/
	int _lane_x = 150;
	int _lane_y = 10;
	int _lane_width = 300;//<400
	int _lane_height = 170;
	/****face box***********/
	int _face_x=350;
	int _face_y=1026;
	int _face_width=400;
	int _face_height=480;
	/****peo_num box***********/
	int _peo_num_x=340;
	int _peo_num_y=910;
	int _peo_num_width=1056;
	int _peo_num_height=528;
	/****lane information***********/
	float _left_point_x= 70;
	float _left_point_y=115.5;
	float _left_angle=-64.1215897;
	float _right_point_x= 217.75;
	float _right_point_y=54.25;
	float _right_angle=64.1731491;
	float _double_lane_dis=135.0;
	float _change_angle=15.0;
	float _left_turn = 0.2;
	float _right_turn = 0.7;
	float _left_weight;//Ô¤Áô
	float _left_bias;//Ô¤Áô
	float _right_weight;//Ô¤Áô
	float _right_bias;//Ô¤Áô
};

void camera(cv::Mat frame_full);
void face_camera(cv::Mat frame_full);
void peonum_camera(cv::Mat frame_full);
void lane_camera_box(cv::Mat frame_full);