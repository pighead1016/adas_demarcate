#include "stdafx.h"
#include "camera_calibration.h"
#include <seeta/FaceDetector.h>
#include <seeta/FaceLandmarker.h>

#include <seeta/Struct_cv.h>
#include <seeta/Struct.h>
adas_camera jp6_camera;
float nx, ny;
bool face_camera_bool = true;
int face_num_seeta = 0;
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
seeta::ModelSetting::Device device = seeta::ModelSetting::CPU;
int id = 0;
seeta::ModelSetting FD_model("./fd_2_00.dat", device, id);
seeta::ModelSetting FL_model("./pd_2_00_pts81.dat", device, id);

seeta::FaceDetector FD(FD_model);
seeta::FaceLandmarker FL(FL_model);
void face_camera(cv::Mat frame_full,int seeta_bool=0)
{
	face_camera_bool = false;
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
	Mat mask=Mat::zeros(Size(400, 480), CV_8UC1);
	face_mask_show = 255 - face_mask_show;
	face_mask_show.copyTo(mask(Rect((mask.cols-face_mask_show.cols)/2, 40, face_mask_show.cols, face_mask_show.rows)));
		//warpAffine(face_mask_show, rot_img, rotate, Size(face_mask_show.cols, face_mask_show.rows));
	std::vector<std::vector<cv::Point>> p; 
	cv::findContours(mask, p, 0, 2);
	cvtColor(mask, mask, CV_GRAY2BGR);

	Mat add_mask_show, rotate_mat;

	resize(mask, mask, face_temp.size());
	
	if(seeta_bool){
		seeta::cv::ImageData simage = face_temp;
		
		auto faces = FD.detect(simage);
		face_num_seeta = faces.size;
		if (faces.size == 1)
		{
			auto &face = faces.data[0];
			auto points = FL.mark(simage, face.pos);
			float det_th = atan2f(points[9].y - points[0].y, points[9].x - points[0].x) * 180 / M_PI;
			std::vector<cv::Point2d> rotate_p(81);
			int p_num = 0;
			Mat rotate = getRotationMatrix2D(Point(points[34].x, points[34].y), det_th, 1) ;
			Mat full_rotate = (Mat_<double>(3, 3) << 0, 0, 0, 0, 0, 0, 0, 0, 1);
			rotate.copyTo(full_rotate.rowRange(0, 2));
			Mat inv_rotate=full_rotate.inv();
			warpAffine(face_temp, rotate_mat, rotate, face_temp.size());
			for (auto &point : points)
			{
				Mat p = (Mat_<double>(3, 1) << point.x, point.y, 1.0);
				Mat rop = rotate*p;//cv::circle(frame, cv::Point(point.x, point.y), 2, CV_RGB(128, 255, 128), -1);
				rotate_p[p_num].x = rop.at<double>(0);
				rotate_p[p_num].y = rop.at<double>(1);

				//cv::circle(rotate_mat, rotate_p[p_num], 2, CV_RGB(128, 255, 128), -1);
				p_num++;
			}
			





			Rect rect_face = cv::Rect(face.pos.x, face.pos.y, face.pos.width, face.pos.height);
			Point2f mid((rotate_p[36].x + rotate_p[37].x) / 2, (rotate_p[36].y + rotate_p[37].y) / 2);
			Point2f mouse_mid((rotate_p[46].x + rotate_p[47].x) / 2, (rotate_p[46].y + rotate_p[47].y) / 2);
			int x = rect_face.x + rect_face.width / 2;
			int y = rect_face.y + rect_face.height / 2;
			nx = (x - points[35].x) / rect_face.width;
			ny = (y - points[35].y) / rect_face.height;
			//rectangle(face_temp, rect_face, Scalar(0, 0, 255), 3);
			//circle(face_temp, Point(points[35].x, points[35].y), 2, Scalar(0, 0, 255), -1);
			float nose_y = ((points[34].y - rect_face.y) / rect_face.height - 0.5)*0.8;
			float mid_diff = (mid.x - rotate_p[62].x) / (rotate_p[63].x - rotate_p[62].x);
			float phone_w = rotate_p[63].x - rotate_p[62].x;
			float phone_h = (mouse_mid.y - mid.y)*1.2;
			float wdiff = 0.5;
			float w = 1;
			float det_w = fabs(mid_diff - 0.5);
			Rect smone_rect;
			if (mid_diff > 0.5)
			{
				smone_rect.x = rotate_p[62].x + ((mid_diff - 0.5)*1.5 - w)*phone_w;//left
				smone_rect.y = (rotate_p[46].y + rotate_p[47].y) / 2 - nose_y*rect_face.height - phone_h*0.8;
				smone_rect.width = rotate_p[63].x + phone_w*(w + det_w / 2) - smone_rect.x;
				smone_rect.height = phone_h * 2.8;
			}
			else
			{
				smone_rect.x = rotate_p[62].x - ((0.5 - mid_diff) * 0 + w)*phone_w;
				smone_rect.y = (rotate_p[46].y + rotate_p[47].y) / 2 - nose_y*rect_face.height - phone_h*0.8;
				smone_rect.width = rotate_p[63].x - (0.5 - mid_diff)*1.5*phone_w + phone_w*(w + det_w) - smone_rect.x;
				smone_rect.height = phone_h * 2.8;
			}
			rectangle(rotate_mat, smone_rect, cv::Scalar(255, 0, 255), 4);
			Mat rect_p = (Mat_<double>(4,3) << 
				smone_rect.x, smone_rect.y ,1, 
				smone_rect.x, smone_rect.y + smone_rect.height,1,
				smone_rect.x + smone_rect.width, smone_rect.y + smone_rect.height,1, 
				smone_rect.x + smone_rect.width, smone_rect.y , 1);
			Mat rotate_point = inv_rotate*rect_p.t();
			//imshow("face", rotate_mat);
			
			for (int i = 0; i < 4; i++) {
				Point p1 = Point(rotate_point.at<double>(0, i), rotate_point.at<double>(1, i));
				Point p2 = Point(rotate_point.at<double>(0, (i + 1) % 4), rotate_point.at<double>(1, (i + 1) % 4));
				line(face_temp,p1 , p2, Scalar(200, 0, 200), 4);
			}
			//imshow("ro_face", face_temp);
		}
		
	}
	addWeighted(face_temp, 0.5, mask, 0.5, 0, add_mask_show);
	cv::imshow("face_show", add_mask_show);
	//cv::imshow("face_show", face_temp);
	cv::waitKey(1);
	face_camera_bool = true;
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
	
	//imshow("salfsd", line);
	//cv::Mat line = gray(cv::Rect(jp6_camera._lane_x, jp6_camera._lane_y, jp6_camera._lane_width, jp6_camera._lane_height));
	//cv::imshow("bug",line);
	cv::resize(line, line,cv::Size(300, 170));
	//cv::imwrite("bug.jpg", line);
	linefind(line);

}