#include "stdafx.h"
#include "camera_calibration.h"

adas_camera jp6_camera;


void camera(cv::Mat frame_full)
{
	cv::Mat small;
	cv::circle(frame_full, cv::Point(jp6_camera._center_x, jp6_camera._center_y), 816, cv::Scalar(255, 255, 0), 4);
	cv::circle(frame_full, cv::Point(jp6_camera._center_x, jp6_camera._center_y), jp6_camera._radius, cv::Scalar(0, 255, 0), 4);
	cv::resize(frame_full, small, cv::Size(frame_full.rows / 2, frame_full.cols / 2));
	cv::imshow("frame_full", small);
	cv::waitKey(1);
}
Mat face_temp;
void face_camera(cv::Mat frame_full)
{
	
	cv::Rect rect(jp6_camera._face_x, jp6_camera._face_y, jp6_camera._face_width, jp6_camera._face_height);
	cv::Mat face_show;
	int aaaa = 0;
	if (jp6_camera._face_x + jp6_camera._face_width >= 1632 || jp6_camera._face_y + jp6_camera._face_height >= 1632)
		aaaa = 3;
	flip(frame_full(rect), face_temp, 0);
	/*int ran = 20;
	for (int c = 0; c < 20; c++) {
		char num[20];
		sprintf_s(num, "%d", c);
		
		line(face_show, Point(face_show.cols*c / ran, 0), Point(face_show.cols*c / ran, face_show.rows), Scalar(128, 128, 128), 2);
		putText(face_show, num, Point(face_show.cols*(c - 0.3) / ran, 20), 1, 1, Scalar(255, 255, 255));
	}*/
	Mat face_mask_show=imread("mask.png",0);
	cv::threshold(face_mask_show, face_mask_show, 0, 255, 8);
	
	
	
	resize(face_mask_show, face_mask_show, Size(245, 280));
	flip(face_mask_show, face_mask_show, 1);
	Mat mask=Mat::zeros(Size(420, 480), CV_8UC1);
	face_mask_show = 255 - face_mask_show;
	face_mask_show.copyTo(mask(Rect((mask.cols-face_mask_show.cols)/2, 40, face_mask_show.cols, face_mask_show.rows)));
		//warpAffine(face_mask_show, rot_img, rotate, Size(face_mask_show.cols, face_mask_show.rows));
	std::vector<std::vector<cv::Point>> p; 
	cv::findContours(mask, p, 0, 2);
	cvtColor(mask, mask, CV_GRAY2BGR);

	Mat add_mask_show;
	resize(mask, mask, face_temp.size());
	addWeighted(face_temp, 0.5, mask, 0.5, 0, add_mask_show);
	cv::imshow("face_show", add_mask_show);

	//cv::imshow("face_show", face_temp);
	cv::waitKey(1);
}

void peonum_camera(cv::Mat frame_full)
{
	cv::Rect rect(jp6_camera._peo_num_x, jp6_camera._peo_num_y, jp6_camera._peo_num_width, jp6_camera._peo_num_height);
	cv::Mat peo_show;
	flip(frame_full(rect), peo_show, 0);
	//cv::resize(frame_full, frame_full, cv::Size(frame_full.rows / 2, frame_full.cols / 2));
	if (peo_show.cols > 1000)
	{
		cv::resize(peo_show, peo_show, cv::Size(peo_show.cols / 2, peo_show.rows / 2));
	}
	cv::imshow("peo_show", peo_show);
	cv::waitKey(1);
}

void lane_camera_box(cv::Mat frame_full)
{
	lane_init();
	
	
	cv::Mat gray;
	cv::cvtColor(frame_full, gray, COLOR_BGR2GRAY);
	IplImage* showImg = cvCreateImage(cvSize(jp6_camera._lane_width, jp6_camera._lane_height), 8, 1);
	IplImage temp_img(gray);
	//printf("Undistline duration:%ld ms\n", (time_end.tv_sec-time_start.tv_sec) * 1000 + (time_end.tv_nsec-time_start.tv_nsec)/1000000);
	//Undistline(&temp_img, showImg);
	//Undist(&temp_img, showImg, jp6_camera._lane_width, jp6_camera._lane_height, jp6_camera._lane_x, jp6_camera._lane_y);
	//Undistall(&temp_img, showImg);
	Undist_line(&temp_img, showImg, jp6_camera._lane_width, jp6_camera._lane_height, jp6_camera._lane_x, jp6_camera._lane_y);
	cv::Mat line = cvarrToMat(showImg);
	/*cv::resize(line_all, line_all, cv::Size(videoWidth / 2, videoHeight / 2));
	cv::flip(line_all.rowRange(videoHeight / 4, videoHeight / 2), line_all.rowRange(videoHeight / 4, videoHeight / 2), 1);
	cv::Mat line_rectage(Size(videoWidth, videoHeight / 4),CV_8UC3);
	line_all.rowRange(0, videoHeight / 4).copyTo(line_rectage.colRange(0, videoWidth / 2));
	line_all.rowRange(videoHeight / 4, videoHeight / 2).copyTo(line_rectage.colRange(videoWidth / 2, videoWidth));*/
	
	imshow("salfsd", line);
	//cv::Mat line = gray(cv::Rect(jp6_camera._lane_x, jp6_camera._lane_y, jp6_camera._lane_width, jp6_camera._lane_height));
	//cv::imshow("bug",line);
	cv::resize(line, line,cv::Size(300, 170));
	cv::imwrite("bug.jpg", line);
	linefind(line);

}