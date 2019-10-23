#include "line.h"
#include "stdafx.h"
#include "Dstmap.h"
#include "camera_calibration.h"
//#define drawleftline 0
using namespace cv;
using namespace std;

//click left right////
int left_c = -1;//num
int right_c = -1;
float change_angle，left_shouder, right_shouder, left_nose, right_nose;
bool left_lock = false, right_lock = false;//click
extern 
#define _left_miss 20
#define _right_miss 20
#define _left_angle_range 20
#define _right_angle_range 20
#define drawleftline

#define _2lane_width 135
#define _2lane_range 10
bool line_left_angle_flag = false, line_right_angle_flag = false, line_left_turn_end = true, line_right_turn_end = true, line_first_flag = true;
int line_leftfilter, line_rightfilter, line_warn_num, line_left_num, line_right_num;
int left_warning_num = 0;
int right_warning_num = 0;
float line_left_standard, line_right_standard;
typedef std::vector<cv::Vec4i> linesType;
extern adas_camera jp6_camera;
struct line_rorect
{
	float x, y, angle;
	Point2f p1, p2;
};
struct lane_state
{
	bool state;
	line_rorect line_data;
};
double pic_mean;
vector<lane_state> filtered_lines;
//IplImage* showImg = cvCreateImage(cvSize(line_width, line_height), 8, 3); // ��ʼ����ʾͼ��
KalmanFilter KFl(6, 3, 0);
KalmanFilter KFr(6, 3, 0);
Mat measurementl(3, 1, CV_32F);
Mat measurementr(3, 1, CV_32F);
//Mat standright = (Mat_<float>(6, 1) << 54.7, 253.8, 91.79, 0, 0, 0);
Mat standright = (Mat_<float>(6, 1) << 57.9, 200.839, 94.565, 0, 0, 0);
//Mat standleft = (Mat_<float>(6, 1) << -57.21, 131.33, 38.14, 0, 0, 0);
Mat standleft = (Mat_<float>(6, 1) << -61.189, 98.135, 33.299, 0, 0, 0);

void  GammaTransform(cv::Mat image, cv::Mat &dist)
{

	Mat imageGamma;
	//�Ҷȹ�һ��
	image.convertTo(imageGamma, CV_64F, 1.0 / 255, 0);

	//٤��任
	double gamma = 1.2;

	pow(imageGamma, gamma, dist);//dist Ҫ��imageGamma����ͬ����������
								 //normalize(dist, dist, 0, 255, CV_MINMAX);
	dist.convertTo(dist, CV_8U, 255, 0);
}

float line_x(line_rorect l1, int y_row)//the line'x at the y_row
{
	return l1.x + (y_row - l1.y) / tanf(l1.angle / 180 * PI);
}

float l2l_angle(line_rorect l1, line_rorect l2)
{
	float line1_up = line_x(l1, 0);
	float line1_down = line_x(l1, 160);
	float line2_up = line_x(l2, 0);
	float line2_down = line_x(l2, 160);
	float t= (fabs(line1_up - line2_up) + fabs(line1_down - line2_down))*fabs(sinf(l1.angle / 180 * PI));
	return t;

}
bool l2l_distance(line_rorect l1, line_rorect l2)
{
	return sqrtf((l2.x - l1.x)*(l2.x - l1.x) + (l2.y - l1.y)* (l2.y - l1.y))<12.5;
}
int right_need_predict = 5;
int left_need_predict = 5;
Mat predictionr, predictionl;
line_rorect hough_left_init = { 70, 115.5, -60 }, hough_right_init = { 217.75, 54.25, 60 }, hough_left, hough_right;

void min2line(vector<Point>contours, Vec4f & line_state)
{
	vector<float> line_x, line_y;
	float xx = 0, xy = 0, x_tal = 0, y_tal = 0;
	for (size_t p = 0; p < contours.size(); p++)
	{
		xx += contours[p].y*contours[p].y;
		xy += contours[p].x*contours[p].y;
		line_x.push_back(contours[p].y);
		line_y.push_back(contours[p].x);
		x_tal += contours[p].y;
		y_tal += contours[p].x;

	}
	float average_x = x_tal / contours.size();
	float average_y = y_tal / contours.size();
	float _line_k = (xy - contours.size()*average_x*average_y) / (xx - contours.size()*average_x*average_x);
	float _line_b = average_y - _line_k*average_x;
	line_state[0] = average_x;
	line_state[1] = average_y;
	line_state[2] = _line_k;
	line_state[3] = _line_b;
}
bool point_x_sort(Point p1, Point p2)
{
	return p1.x > p2.x;
}
void min2line_has_r2(vector<Point>contours, Vec4f & line_state, float& r2)
{
	Vec4f _fitline;
	//fitLine(contours, _fitline, DIST_L12, 0, 1e-2, 1e-2);
	vector<float> line_x, line_y;
	line_x.clear();
	line_y.clear();
	float xx = 0, xy = 0, x_tal = 0, y_tal = 0;
	sort(contours.begin(), contours.end(), point_x_sort);
	//x��ͬȡ��ֵ
	for (size_t p = 0; p < contours.size(); ++p)
	{
		float sum_y = contours[p].y;
		int sim_num = 1;
		size_t q;
		for (q = p + 1; q < contours.size(); q++)
		{
			if (contours[p].x == contours[q].x)
			{
				sum_y += contours[q].y;
				sim_num++;
				continue;
			}
			else
			{
				for (size_t k = p; k < q; k++)
				{
					line_x.push_back(contours[p].x);
					line_y.push_back(sum_y / (q - p));
					//contours[k].y = sum_y / (q - p);
				}
				p = q - 1;
				break;
			}
		}

		if (q == contours.size())//���һ�����
		{
			for (int k = 0; k < sim_num; k++)
			{
				line_x.push_back(contours[p].x);
				line_y.push_back(sum_y / sim_num);
			}
			break;
		}
		else if (p == contours.size() - 2) {
			line_x.push_back(contours[p].x);
			line_y.push_back(contours[p].y);
			break;
		}
	}
	for (size_t p = 0; p < contours.size(); p++)
	{
		xx += line_x[p] * line_x[p];
		xy += line_x[p] * line_y[p];

		x_tal += line_x[p];
		y_tal += line_y[p];

	}
	float average_x = x_tal / contours.size();
	float average_y = y_tal / contours.size();
	float _line_k = (xy - contours.size()*average_x*average_y) / (xx - contours.size()*average_x*average_x);
	float _line_b = average_y - _line_k*average_x;
	line_state[0] = average_x;
	line_state[1] = average_y;
	line_state[2] = _line_k;
	line_state[3] = _line_b;
	float ssr = 0, sst = 0;
	for (size_t p = 0; p < contours.size(); p++)
	{
		float y_n = line_x[p] * _line_k + _line_b;
		ssr += (y_n - line_state[1])*(y_n - line_state[1]);
		sst += (line_y[p] - line_state[1])*(line_y[p] - line_state[1]);
	}
	r2 = ssr / sst;
}
Point2f line_center;
bool ppt(Point p1, Point p2)
{
	return (p1.x*line_center.x + line_center.y - p1.y) > (p2.x*line_center.x + line_center.y - p2.y);
}
struct min2line_Struct
{
	Vec4f line_state;
	float r2;
};
float minangle_r = 50;
float minangle_l = 40;
void fit_line(Mat drawline, vector<Point>contours, float box_scal, Vec4f line_state, vector<min2line_Struct>& r2_and_state)
{
	//vector<Point> uppoint, downpoint;
	
	line_center = Point2f(line_state[2], line_state[3]);
	sort(contours.begin(), contours.end(), ppt);
	int zero_num = 0;
	for (size_t i = 1; i < contours.size(); i++)
	{
		if (contours[i].x*line_state[2] + line_state[3] - contours[i].y < 0)
		{
			zero_num = i;
			break;
		}
	}
	int up_num = zero_num*box_scal;
	int down_num = (contours.size() - zero_num)*box_scal;

	vector<Point> uppoint(contours.begin(), contours.begin() + up_num);
	vector<Point> downpoint(contours.rbegin(), contours.rbegin() + down_num);
	for (size_t i = 0; i < uppoint.size(); i++)
	{
		circle(drawline, uppoint[i], 1, Scalar(255, 128, 255));
	}
	for (size_t i = 0; i < downpoint.size(); i++)
	{
		circle(drawline, downpoint[i], 1, Scalar(128, 255, 255));
	}
	//imshow("pp", drawline);

	min2line_Struct up_line;
	min2line_has_r2(uppoint, up_line.line_state, up_line.r2);
	r2_and_state[0] = up_line;
	min2line_Struct down_line;
	min2line_has_r2(downpoint, down_line.line_state, down_line.r2);
	r2_and_state[1] = down_line;
	//cout << " : r2=" << r2 << endl;

}
vector<Point2f> two_line2_four_point(vector<min2line_Struct> r2_and_state, double long_line)
{
	vector<Point2f> point4;
	for (size_t i = 0; i < r2_and_state.size(); ++i)
	{
		float angle = atanf(r2_and_state[i].line_state[2]);
		float x0, x1, y0, y1;
		x0 = r2_and_state[i].line_state[0] + long_line / 2 * cos(angle);
		x1 = r2_and_state[i].line_state[0] - long_line / 2 * cos(angle);
		y0 = r2_and_state[i].line_state[1] + long_line / 2 * sin(angle);
		y1 = r2_and_state[i].line_state[1] - long_line / 2 * sin(angle);
		point4.push_back(Point2f(x0, y0));

		point4.push_back(Point2f(x1, y1));
	}
	return point4;
}
bool draw2line(Mat drawimg, vector<min2line_Struct> r2_and_state, double area, double height, double width, line_rorect & cur_line)
{
	vector<Point2f> point4 = two_line2_four_point(r2_and_state, max(height, width));
	for (size_t i = 0; i < point4.size(); i++)
	{
		circle(drawimg, point4[i], 1, Scalar(0, 0, 128));
	}
	vector<Point2f> point4_new;
	convexHull(point4, point4_new);
	double point_4_area = contourArea(point4_new);
	//cout << "area prent " << area / point_4_area ;

	float y0 = (r2_and_state[0].line_state[1] + r2_and_state[1].line_state[1]) / 2;
	float x0 = (r2_and_state[0].line_state[0] + r2_and_state[1].line_state[0]) / 2;
	cur_line.x = x0;
	cur_line.y = y0;
	//printf("  r2 --> line1= %3.2f line2=%3.2f,center(%1.1f,%1.1f)\n",r2_and_state[0].r2, r2_and_state[1].r2, r2_and_state[0].line_state[0], r2_and_state[0].line_state[1]);
	if (area / point_4_area < 0.9)
		return false;
	if (r2_and_state[0].r2 > 0.85&&r2_and_state[1].r2 > 0.85)
	{
		float tan1 = r2_and_state[0].line_state[2];
		float tan2 = r2_and_state[1].line_state[2];
		float tan_diff = fabs((tan1 - tan2) / (1 + tan1*tan2));

		float max_diff = 0.15 - fabs(x0 - 160) / 160 * (0.15 - 0.05);
		if (tan_diff < max_diff)
		{
			float _k_ = (atanf(tan1) + atanf(tan2)) / 2;
			if (tan1*tan2 < 0)
				_k_ -= M_PI_2;
			if (width > height)
			{
				if (_k_ > 0)
					cur_line.angle = _k_ / M_PI * 180 - 180;
				else
					cur_line.angle = _k_ / M_PI * 180;
			}
			else
			{
				if (_k_ > 0)
					cur_line.angle = _k_ / M_PI * 180;
				else
					cur_line.angle = _k_ / M_PI * 180 + 180;
			}
			//float k_average = tan(_k_);
			//int line_x_up = x0 - y0 / k_average;
			//int line_x_down = x0 + (drawimg.rows - y0) / k_average;
			return true;
			//if(color==1)
			//line(drawimg, Point(line_x_up, 0), Point(line_x_down, drawimg.rows), Scalar(255, 128, 128), 2);
		}
		else
		{
			//cout << "not simlar " << tan_diff << endl;
			return false;
		}
	}
	else
	{
		int which_line = -1;
		if (r2_and_state[0].r2 > 0.9)
			which_line = 0;
		else if (r2_and_state[1].r2 > 0.9)
			which_line = 1;
		if (which_line != -1)
		{
			//cout << "only one line" << endl;
			//int line_x_up = y0 - r2_and_state[which_line].line_state[2] * x0;
			//int line_x_down = y0 + r2_and_state[which_line].line_state[2] * (drawimg.cols - x0);
			//line(drawimg, Point(0, line_x_up), Point(drawimg.cols, line_x_down), Scalar(255, 128, 128), 2);
			float _k_ = atanf(r2_and_state[which_line].line_state[2]);
			if (width > height)
			{
				if (_k_ > 0)
					cur_line.angle = _k_ / M_PI * 180 - 180;
				else
					cur_line.angle = _k_ / M_PI * 180;
			}
			else
			{
				if (_k_ > 0)
					cur_line.angle = _k_ / M_PI * 180;
				else
					cur_line.angle = _k_ / M_PI * 180 + 180;
			}

			return true;
		}
		return false;
	}
}
line_rorect true_right_line;
line_rorect true_left_line;
#define stand_right 65.1731491
#define stand_left -65.1215897
#define change_angle 7.5
void getROI(cv::Mat img, std::vector<cv::Point> vertices, cv::Mat& masked,float &mask_mean) {
	cv::Mat mask = cv::Mat::zeros(img.size(), img.type());
	if (img.channels() == 1) {
		cv::fillConvexPoly(mask, vertices, cv::Scalar(255));
	}
	else if (img.channels() == 3) {
		cv::fillConvexPoly(mask, vertices, cv::Scalar(255, 255, 255));
	}
	mask_mean = mean(mask)[0] / 255.0;
	cv::bitwise_and(img, img, masked, mask);
}
void drawlrLines(cv::Mat& img, vector<lane_state> lrline) {
	if (lrline.at(0).state) {
		float leftx = line_x(lrline.at(0).line_data, 0);
		float leftdx = line_x(lrline.at(0).line_data, img.rows);
		line(img, Point(leftx, 0), Point(leftdx, img.rows), Scalar(0, 0, 255),4);
	}
	if (lrline.at(1).state) {
		float rightx = line_x(lrline.at(1).line_data, 0);
		float rightdx = line_x(lrline.at(1).line_data, img.rows);
		line(img, Point(rightx, 0), Point(rightdx, img.rows), Scalar(0, 255, 0),4);
	}
	
}
void drawLines_single(cv::Mat& img, vector<line_rorect> lrline) {
	for (size_t i = 0; i<lrline.size(); i++) {
		//float rightx = line_x(lrline[i].first, 0);
		line(img, lrline[i].p1, lrline[i].p2, Scalar(200,0,100), 2);
		
	}

}
void drawLines_pair(cv::Mat& img, vector< line_rorect>  lrline,Scalar color) {
	for (size_t i = 0; i < lrline.size(); i++) {
		//float rightx = line_x(lrline[i].first, 0);
			//line(img, lrline[i].first.p1, lrline[i].first.p2, color, 4);
		char name[20];
		sprintf_s(name, "%d", i+1);
		
		line(img, lrline[i].p1 ,  lrline[i].p2, color, 2);
		putText(img, name, lrline[i].p2, 1, 1, Scalar(255, 255, 255) - color, 2);
	}
	

}
void drawLines(cv::Mat& img, linesType lines, cv::Scalar color) {
	for (size_t i = 0; i < lines.size(); ++i) {
		cv::Point pt1 = cv::Point(lines[i][0], lines[i][1]);
		cv::Point pt2 = cv::Point(lines[i][2], lines[i][3]);
		cv::line(img, pt1, pt2, color, 1, 8);
	}
}
float l2r_lane(line_rorect l_lane, line_rorect r_lane)
{
	return line_x(r_lane, 80) - line_x(l_lane, 80);
}
struct lane_info
{
	float socer;
	line_rorect lane_data;
};
bool lane_socer(lane_info l1, lane_info l2) {
	return l1.socer < l2.socer;
}
float p2line(line_rorect l1, Point2f p)//distance of point to line
{
	float k = tan(l1.angle / 180 * PI);
	return (k*p.x - p.y + l1.y - k*l1.x) / sqrt(k*k + 1);
}
float fit_width(float y)
{
	return y*0.037377 + 5.8484;
}
float Roi_mean(Mat img, line_rorect l1, line_rorect l2)
{
	vector<Point> points{ l1.p1,l1.p2,l2.p2,l2.p1 };
	Mat maskroi;
	float me;
	getROI(img, points, maskroi,me);
	//imshow("roi", maskroi);
	//waitKey(1);
	float area = contourArea(points);
	float roi_mean = mean(maskroi)[0] / me;
	return roi_mean;
}
float fit_left(float y)
{
	return 0.0462512727849216f*y + 8.28599905275439f;
}
float fit_right(float y)
{
	return 0.0374693487874771f*y + 9.509242117f;
}
bool cmp_temp(pair<int, line_rorect> a, pair<int, line_rorect> b)
{
	return a.first<b.first;//根据fisrt的值升序排序
}
void concat_line(vector<line_rorect>src_line,vector<line_rorect>&dst_line)
{
	vector<bool> in_team(src_line.size(), false);
	vector<pair<int, line_rorect> > line_team;
	int line_num = int(src_line.size());
	if (line_num<2)
	{
		return ;
	}
	for (int i = 0; i <line_num - 1; i++)
	{
		if (in_team[i])
			continue;
		line_team.push_back(make_pair(i, src_line[i]));
		for (int j = i+1; j < line_num; j++)
		{
			if (in_team[j])
				continue;
			float p1d = abs(p2line(src_line[i], src_line[j].p1));
			//float p1dfit = fit_right(src_line[j].p1.y);
			float p2d = abs(p2line(src_line[i], src_line[j].p2));
			//float p2dfit = fit_right(src_line[j].p2.y);
			//float angle_diff = l2l_angle(src_line[i], src_line[j]);
			//if (angle_diff < 15)//same line
			if(p1d<5&&p2d<5)
			{
				line_team.push_back(make_pair(i, src_line[j]));
				in_team[j] = true;
			}
		}
	}
	if (!in_team[line_num - 1])
	{
		line_team.push_back(make_pair(line_num - 1, src_line[line_num - 1]));
	}

	sort(line_team.begin(), line_team.end(), cmp_temp);
	
	line_rorect totle = line_team[0].second;
	int same_num = 1;
	for (int i = 0; i < line_num - 1; i++)
	{
		if (line_team[i].first == line_team[i + 1].first)//same line
		{
			same_num++;
			totle.angle+=line_team[i + 1].second.angle;
			totle.x += line_team[i + 1].second.x;
			totle.y += line_team[i + 1].second.y;
			if(totle.p1.y > line_team[i + 1].second.p1.y)
				totle.p1= line_team[i + 1].second.p1;
			if (totle.p2.y < line_team[i + 1].second.p2.y)
				totle.p2 = line_team[i+1].second.p2;
		}
		else
		{
			line_rorect t;
			t.angle = totle.angle / same_num;
			t.x = totle.x / same_num;
			t.y = totle.y / same_num;
			t.p1 = totle.p1;
			t.p2 = totle.p2;
			dst_line.push_back(t);
			totle = line_team[i + 1].second;
			same_num = 1;
		}
	}
	line_rorect t;
	t.angle = totle.angle / same_num;
	t.x = totle.x / same_num;
	t.y = totle.y / same_num;
	t.p1 = totle.p1;
	t.p2 = totle.p2;
	dst_line.push_back(t);
}
float p2p_dis(line_rorect l)
{
	float x = l.p1.x - l.p2.x;
	float y = l.p1.y - l.p2.y;
	return sqrt(x*x + y + y);
}
void line_widthfilter(vector<line_rorect>&lines, vector<line_rorect>&lanes,float tofit)
{
	if (lines.size() < 2)
		return;
	for (int i = 0; i < lines.size() - 1; i++)
	{
		for (int j = i + 1; j < lines.size(); j++)
		{
			int l = j;
			int s = i;
			//if (p2p_dis(lines[i]) > p2p_dis(lines[j]))
			//{
			//	l = i;
			//	s = j;
			//}
			float p1d = p2line(lines[l], lines[s].p1);
			float p2d = p2line(lines[l], lines[s].p2);
			
			if (p1d*p2d <= 0)//filter line with points at different side
				continue;
			//std::cout << "left:" << lines[i].angle<<" " << lines[j].p1.y << " " << p1d << " " << lines[j].p2.y << " " << p2d << endl;
			float p1dfit = fit_right(lines[s].p1.y);
			p1d = fabs(p1d);
			p2d = fabs(p2d);
			float scale;
			if (p1dfit > p1d)
				scale = p1dfit / p1d;
			else
				scale = p1d / p1dfit;
			//cout << " scale1 " << scale << endl;
			if (scale > tofit)
				continue;
			float p2dfit = fit_right(lines[s].p2.y);
			if (p2dfit > p2d)
				scale = p2dfit / p2d;
			else
				scale = p2d / p2dfit;
			//cout << " scale2 " << scale << endl;

			if (scale > tofit)
				continue;
			line_rorect t;
			t.angle = (lines[i].angle + lines[j].angle) / 2;
			t.x= (lines[i].x + lines[j].x) / 2;
			t.y= (lines[i].y + lines[j].y) / 2;
			t.p1= (lines[i].p1 + lines[j].p1) / 2;
			t.p2 = (lines[i].p2 + lines[j].p2) / 2;
			lanes.push_back(t);
		}
	}
}
void bypassangle_linewidth(linesType lines, line_rorect left_predict_line, line_rorect right_predict_line,vector< line_rorect>&left_lanes, vector< line_rorect >&right_lanes)
{
	std::vector<line_rorect> left_line, right_line;
	std::vector<line_rorect> long_left_line, long_right_line;
	for (size_t i = 0; i < lines.size(); ++i) {
		float x1 = lines[i][0], y1 = lines[i][1];
		float x2 = lines[i][2], y2 = lines[i][3];
		float angle = atan((y2 - y1) / (x2 - x1)) * 180 / PI;
		float x = (x1 + x2) / 2;
		float y = (y1 + y2) / 2;
		float length = sqrt((x1 - x2)*(x1 - x2) + (y1 - y2)*(y1 - y2));
		Point2f upp, downp;
		if (y1 > y2) {
			upp = Point2f(x2, y2);
			downp = Point2f(x1, y1);
		}
		else {
			upp = Point2f(x1, y1);
			downp = Point2f(x2, y2);
		}
		line_rorect temp_lane{ x,y,angle,upp,downp };
		if (fabs(angle - left_predict_line.angle) < _right_angle_range) {
			left_line.push_back(temp_lane);
		}
		if (fabs(angle - right_predict_line.angle) < _left_angle_range) {
			right_line.push_back(temp_lane);
		}
	}
	//Mat co = Mat::zeros(Size(300, 170), CV_8UC3);
	//cout << "left" << endl;
	concat_line(left_line,long_left_line);
	//drawLines_single(co, long_left_line);
 	line_widthfilter(long_left_line, left_lanes, 2);
	//cout << "right" << endl;
	concat_line(right_line, long_right_line);
	line_widthfilter(long_right_line, right_lanes, 2);/////////renew lane with width not color
	//drawLines_single(co, long_right_line);
	//imshow("long", co);
}
void bypass_color(Mat img, vector<pair<line_rorect, line_rorect> >lanes,vector<line_rorect>& white_lane)
{
	vector<float> socer;
	for (size_t i = 0; i < lanes.size(); i++)
	{
		float mean2line = Roi_mean(img, lanes[i].first, lanes[i].second);
		socer.push_back(mean2line);
	}
	float k_socer;
	vector<int> label;
	vector<float> cen;
	Mat outcenter,mean,std;
	meanStdDev(socer, mean, std);
	if (std.at<double>(0) < 10)//same gray
	{
		if (mean.at<double>(0) > pic_mean)//all right
		{
			for (size_t i = 0; i < lanes.size(); i++)
			{
				line_rorect t;
				t.angle = (lanes[i].first.angle + lanes[i].second.angle) / 2;
				t.x = (lanes[i].first.x + lanes[i].second.x) / 2;
				t.y = (lanes[i].first.y + lanes[i].second.y) / 2;
				t.p1 = (lanes[i].first.p1 + lanes[i].second.p1) / 2;
				t.p2 = (lanes[i].first.p2 + lanes[i].second.p2) / 2;
				white_lane.push_back(t);
			}
		}
	}
	else {
		if (socer.size() >= 2) {
			k_socer = kmeans(socer, 2, label,
				TermCriteria(TermCriteria::EPS + TermCriteria::MAX_ITER, 10, 0), 2, KMEANS_PP_CENTERS, cen);
			cout << k_socer << "  " << cen[0] << "  " << cen[1] << endl;
			int num = 0;
			if (cen[0] < cen[1])
				num = 1;
			for (size_t i = 0; i < label.size(); i++)
			{
				if (label[i] == num) {
					line_rorect t;
					t.angle = (lanes[i].first.angle + lanes[i].second.angle) / 2;
					t.x = (lanes[i].first.x + lanes[i].second.x) / 2;
					t.y = (lanes[i].first.y + lanes[i].second.y) / 2;
					t.p1 = (lanes[i].first.p1 + lanes[i].second.p1) / 2;
					t.p2 = (lanes[i].first.p2 + lanes[i].second.p2) / 2;
					white_lane.push_back(t);
				}
			}
		}
	}
}
void bypass_laneFilter(vector<line_rorect> white_left_lane, vector<line_rorect> white_right_lane,line_rorect left_predict_line, line_rorect right_predict_line, vector<lane_state>& output)
{
	vector< vector<line_rorect> > white_lanes;
	white_lanes.push_back(white_left_lane);
	white_lanes.push_back(white_right_lane);
	vector<line_rorect> predict_lines;
	predict_lines.push_back(left_predict_line);
	predict_lines.push_back(right_predict_line);
	vector<bool> temp_state{ output.at(0).state ,output.at(1).state };//copy state
	output.at(0).state = false;//clean state
	output.at(1).state = false;//
	int less = 0;
	int more = 1;
	if (white_left_lane.size() > white_right_lane.size())
	{
		less = 1, more = 0;
	}
	float min_fit;
	if(temp_state[less])
		min_fit= 40;
	else
		min_fit = 90;
	for (size_t i = 0; i < white_lanes[less].size(); i++)
	{
		line_rorect temp_l= white_lanes[less][i];
		float lane2par = l2l_angle(temp_l, predict_lines[less]);
		if (lane2par < min_fit)
		{
			min_fit = lane2par;
			output.at(less).state = true;
			output.at(less).line_data = temp_l;
		}
	}
	if (temp_state[more])
		min_fit = 40;
	else
		min_fit = 90;
	float min_to_fit_width = _2lane_range;
	for (size_t i = 0; i < white_lanes[more].size(); i++)
	{
		line_rorect temp_l= white_lanes[more][i];
			
		float dis_2lane;
		if (output.at(less).state == true) {//now found right lane
			dis_2lane = abs(l2r_lane(temp_l, output[less].line_data));
			if (abs(dis_2lane - _2lane_width) > min_to_fit_width)
				continue;
			else {
				min_to_fit_width = abs(dis_2lane - _2lane_width);
				output.at(more).state = true;
				output.at(more).line_data = temp_l;
			}
		}
		else if (temp_state[less]) {//last found (less lanes)side lane
			dis_2lane = abs(l2r_lane(temp_l, predict_lines[less]));
			if (abs(dis_2lane - _2lane_width) > min_to_fit_width)
				continue;
			else {
				min_to_fit_width = abs(dis_2lane - _2lane_width);
				output.at(more).state = true;
				output.at(more).line_data = temp_l;
			}
		}
		else {
			float lane2par = l2l_angle(temp_l, predict_lines[more]);
			if (lane2par < min_fit)
			{
				min_fit = lane2par;
				output.at(more).state = true;
				output.at(more).line_data = temp_l;
			}
		}
	}
}
void bypassAngleFilter(Mat img,linesType lines, line_rorect left_predict_line, line_rorect right_predict_line ,vector<lane_state>& output) {
	std::vector<line_rorect> left_line, right_line;
	std::vector<lane_info> left_lane_info, right_lane_info;
	
	for (size_t i = 0; i < lines.size(); ++i) {
		float x1 = lines[i][0], y1 = lines[i][1];
		float x2 = lines[i][2], y2 = lines[i][3];
		float angle = atan((y2 - y1) / (x2 - x1)) * 180 / PI;
		float x = (x1 + x2) / 2;
		float y = (y1 + y2) / 2;
		float length = sqrt((x1 - x2)*(x1 - x2) + (y1 - y2)*(y1 - y2));
		Point2f upp, downp;
		if (y1 > y2) {
			upp = Point2f(x2, y2);
			downp = Point2f(x1, y1);
		}
		else {
			upp = Point2f(x1, y1);
			downp = Point2f(x2, y2);
		}
		line_rorect temp_lane{ x,y,angle,upp,downp };
		if (fabs(angle - left_predict_line.angle) < _right_angle_range) {
			left_line.push_back(temp_lane);
		}
		if (fabs(angle - right_predict_line.angle) < _left_angle_range) {
			right_line.push_back(temp_lane);
		}
	}
	
	bool temp_l_state = output.at(0).state, temp_r_state = output.at(1).state;
	output.at(0).state = false;
	output.at(1).state = false;
	//float fit_lane = (fit_width(0)+ fit_width(160))*1.2;//线宽
	if (right_line.size() > left_line.size())
	{
		float min_fit_left = 90;

		vector<Point> left_lane;
		if (left_line.size() > 1) {
			for (size_t i = 0; i < left_line.size() - 1; i++)
			{
				for (size_t j = i + 1; j < left_line.size(); j++)
				{
					float mean2line = Roi_mean(img, left_line[i], left_line[j]);
					if (mean2line < pic_mean)//not white inside
						continue;
					float angle_diff = l2l_angle(left_line[i], left_line[j]);
					if (angle_diff == 0)
						continue;
					float p1d = p2line(left_line[i], left_line[j].p1);
					float p2d = p2line(left_line[i], left_line[j].p2);
					//std::cout << "left:" << left_line[i].angle<<" " << left_line[j].p1.y << " " << p1d << " " << left_line[j].p2.y << " " << p2d << endl;
					if (p1d*p2d <= 0)//filter line with points at different side
						continue;
					float p1dfit = fit_right(left_line[j].p1.y);
					p1d = fabs(p1d);
					p2d = fabs(p2d);
					float scale;
					if (p1dfit > p1d)
						scale = p1dfit / p1d;
					else
						scale = p1d / p1dfit;
					if (scale > 1.3)
						continue;
					float p2dfit = fit_right(left_line[j].p2.y);
					if (p2dfit > p2d)
						scale = p2dfit / p2d;
					else
						scale = p2d / p2dfit;
					if (scale > 1.3)
						continue;
					left_lane.push_back(Point(i, j));
				}
			}
			for (size_t i = 0; i < left_lane.size(); i++)
			{
				line_rorect temp_l;
				temp_l.x = (left_line[left_lane[i].x].x + left_line[left_lane[i].y].x) / 2;
				temp_l.y = (left_line[left_lane[i].x].y + left_line[left_lane[i].y].y) / 2;
				temp_l.angle = (left_line[left_lane[i].x].angle + left_line[left_lane[i].y].angle) / 2;

				float lane2par = l2l_angle(temp_l, left_predict_line);
				if (lane2par < min_fit_left)
				{
					min_fit_left = lane2par;
					output.at(0).state = true;
					output.at(0).line_data = temp_l;
				}
			}
		}
		if (right_line.size() > 1) {
			//int right1 = -1, right2 = -1;
			float min_fit_right = 90;
			float min_to_fit_width = 15;

			vector<Point> right_lane;
			for (size_t i = 0; i < right_line.size() - 1; i++)
			{
				for (size_t j = i + 1; j < right_line.size(); j++)
				{
					float mean2line = Roi_mean(img, right_line[i], right_line[j]);
					if (mean2line < pic_mean)//not white inside
						continue;
					//float angle_diff = l2l_angle(right_line[i], right_line[j]);
					//if (angle_diff == 0)
					//	continue;
					float p1d = p2line(right_line[i], right_line[j].p1);
					float p2d = p2line(right_line[i], right_line[j].p2);
					//std::cout << "right:" << right_line[i].angle << " " << right_line[j].p1.y << " " << p1d << " " << right_line[j].p2.y << " " << p2d << endl;
					if (p1d*p2d <= 0)//filter line with points at different side
						continue;
					float p1dfit = fit_right(right_line[j].p1.y);
					p1d = fabs(p1d);
					p2d = fabs(p2d);
					float scale;
					if (p1dfit > p1d)
						scale = p1dfit / p1d;
					else
						scale = p1d / p1dfit;
					if (scale > 1.25)
						continue;
					float p2dfit = fit_right(right_line[j].p2.y);
					if (p2dfit > p2d)
						scale = p2dfit / p2d;
					else
						scale = p2d / p2dfit;
					if (scale > 1.25)
						continue;
					right_lane.push_back(Point(i, j));
				}
			}
			for (size_t i = 0; i < right_lane.size(); i++)
			{
				line_rorect temp_l;
				temp_l.x = (right_line[right_lane[i].x].x + right_line[right_lane[i].y].x) / 2;
				temp_l.y = (right_line[right_lane[i].x].y + right_line[right_lane[i].y].y) / 2;
				temp_l.angle = (right_line[right_lane[i].x].angle + right_line[right_lane[i].y].angle) / 2;
				float r2left_predict;
				if (output.at(0).state == true) {//now found right lane
					r2left_predict = l2r_lane(temp_l, output[0].line_data);
					if (abs(r2left_predict + _2lane_width) > min_to_fit_width)
						continue;
					else {
						min_to_fit_width = abs(r2left_predict + _2lane_width);
						output.at(1).state = true;
						output.at(1).line_data = temp_l;
					}
				}
				else if (temp_l_state) {//last found right lane
					r2left_predict = l2r_lane(temp_l, left_predict_line);
					if (abs(r2left_predict + _2lane_width) > min_to_fit_width)
						continue;
					else {
						min_to_fit_width = abs(r2left_predict + _2lane_width);
						output.at(1).state = true;
						output.at(1).line_data = temp_l;
					}
				}
				else {
					float lane2par = l2l_angle(temp_l, right_predict_line);
					if (lane2par < min_fit_right)
					{
						min_fit_right = lane2par;
						output.at(1).state = true;
						output.at(1).line_data = temp_l;
					}
				}
			}
		}
	}
	else {
		if (right_line.size() > 1) {
			float min_fit_right = 90;
			vector<Point> right_lane;
			for (size_t i = 0; i < right_line.size() - 1; i++)
			{
				for (size_t j = i + 1; j < right_line.size(); j++)
				{
					float mean2line = Roi_mean(img, right_line[i], right_line[j]);
					if (mean2line < pic_mean)//not white inside
						continue;
					float angle_diff = l2l_angle(right_line[i], right_line[j]);
					if (angle_diff == 0)
						continue;
					float p1d = p2line(right_line[i], right_line[j].p1);
					float p2d = p2line(right_line[i], right_line[j].p2);
					if (p1d*p2d <= 0)//filter line with points at different side
						continue;
					float p1dfit = fit_right(right_line[j].p1.y);
					p1d = fabs(p1d);
					p2d = fabs(p2d);
					float scale;
					if (p1dfit > p1d)
						scale = p1dfit / p1d;
					else
						scale = p1d / p1dfit;
					if (scale > 1.25)
						continue;
					float p2dfit = fit_right(right_line[j].p2.y);
					if (p2dfit > p2d)
						scale = p2dfit / p2d;
					else
						scale = p2d / p2dfit;
					if (scale > 1.25)
						continue;
						right_lane.push_back(Point(i, j));
				}
			}
			for (size_t i = 0; i < right_lane.size(); i++)
			{
				line_rorect temp_l;
				temp_l.x = (right_line[right_lane[i].x].x + right_line[right_lane[i].y].x) / 2;
				temp_l.y = (right_line[right_lane[i].x].y + right_line[right_lane[i].y].y) / 2;
				temp_l.angle = (right_line[right_lane[i].x].angle + right_line[right_lane[i].y].angle) / 2;
				//float r2left_predict;

				float lane2par = l2l_angle(temp_l, right_predict_line);
				if (lane2par < min_fit_right)
				{
					min_fit_right = lane2par;
					output.at(1).state = true;
					output.at(1).line_data = temp_l;
				}
			}
		}

		float min_fit_left = 90;
		float min_to_fit_width = 15;

		vector<Point> left_lane;
		if (left_line.size() > 1) {
			for (size_t i = 0; i < left_line.size() - 1; i++)
			{
				for (size_t j = i + 1; j < left_line.size(); j++)
				{
					float mean2line = Roi_mean(img, left_line[i], left_line[j]);
					if (mean2line < pic_mean)//not white inside
						continue;//waitKey(1);

					float angle_diff = l2l_angle(left_line[i], left_line[j]);
					if (angle_diff == 0)
						continue;
					float p1d = p2line(left_line[i], left_line[j].p1);
					float p2d = p2line(left_line[i], left_line[j].p2);
					//std::cout << "left:" << left_line[i].angle << " " << left_line[j].p1.y << " " << left_line[j].p1.x << " " << p1d << endl;
					//std::cout << "left:" << left_line[i].angle << " " << left_line[j].p2.y << " " << left_line[j].p2.x << " " << p2d << endl;
					if (p1d*p2d <= 0)//filter line with points at different side
						continue;
					float p1dfit = fit_right(left_line[j].p1.y);
					p1d = fabs(p1d);
					p2d = fabs(p2d);
					float scale;
					if (p1dfit > p1d)
						scale = p1dfit / p1d;
					else
						scale = p1d / p1dfit;
					if (scale > 1.3)
						continue;
					float p2dfit = fit_right(left_line[j].p2.y);
					if (p2dfit > p2d)
						scale = p2dfit / p2d;
					else
						scale = p2d / p2dfit;
					if (scale > 1.3)
						continue;
						left_lane.push_back(Point(i, j));
				}
			}
			for (size_t i = 0; i < left_lane.size(); i++)
			{
				line_rorect temp_l;
				temp_l.x = (left_line[left_lane[i].x].x + left_line[left_lane[i].y].x) / 2;
				temp_l.y = (left_line[left_lane[i].x].y + left_line[left_lane[i].y].y) / 2;
				temp_l.angle = (left_line[left_lane[i].x].angle + left_line[left_lane[i].y].angle) / 2;
				float l2right_predict;

				if (output.at(1).state == true) {//now found right lane
					l2right_predict = l2r_lane(temp_l, output[1].line_data);
					if (abs(l2right_predict - _2lane_width) > min_to_fit_width)
						continue;
					else {
						min_to_fit_width = abs(l2right_predict - _2lane_width);
						output.at(0).state = true;
						output.at(0).line_data = temp_l;
					}
				}
				else if (temp_r_state) {//last found right lane
					l2right_predict = l2r_lane(temp_l, right_predict_line);
					if (abs(l2right_predict - _2lane_width) > min_to_fit_width)
						continue;
					else {
						min_to_fit_width = abs(l2right_predict - _2lane_width);
						output.at(0).state = true;
						output.at(0).line_data = temp_l;
					}
				}
				else {
					float lane2par = l2l_angle(temp_l, left_predict_line);
					if (lane2par < min_fit_left)
					{
						min_fit_left = lane2par;
						output.at(0).state = true;
						output.at(0).line_data = temp_l;
					}
				}
			}
		}
	}
}
// m: б��, b: �ؾ�, norm: �߳�
struct hough_pts {
	double m, b, norm;
	hough_pts(double m, double b, double norm) :
		m(m), b(b), norm(norm) {};
};
int left_miss_num = _left_miss,right_miss_num=_right_miss;
void refresh_lane(vector<lane_state>filtered_lines, line_rorect& hough_left, line_rorect& hough_right)
{
	if (left_miss_num < _left_miss) {
		if (filtered_lines.at(0).state)
		{
			hough_left = filtered_lines.at(0).line_data;
		}
		else
			left_miss_num++;
	}
	else {
		hough_left = hough_left_init;
		//left_miss_num = 0;
	}
	if (right_miss_num < _right_miss) {
		if (filtered_lines.at(1).state)
		{
			hough_right = filtered_lines.at(1).line_data;
		}
		else
			right_miss_num++;
	}
	else {
		hough_right = hough_right_init;
		//right_miss_num = 0;
	}
}
void lane_init() {
	hough_left = hough_left_init;
	hough_right = hough_right_init;
}
cv::Mat clahe_deal(cv::Mat src)
{
	cv::Mat clahe_img;
	cv::Ptr<cv::CLAHE> clahe = cv::createCLAHE();
	clahe->setClipLimit(4.);    // (int)(4.*(8*8)/256)
	clahe->setTilesGridSize(Size(8, 4));
	clahe->apply(src, clahe_img);
	return clahe_img;
}
bool lane_start = true;
bool in_lane = true;
//int find_line_num=0;
double totle_time=0;
bool save_left = false, save_right = false;

int linefind(Mat findline)
{
	double start = getTickCount();
	//find_line_num++;
	minangle_r = 60;
	minangle_l = 60;
	int turn_dir = 0;
	Mat dstImage= clahe_deal(findline);
	//imshow("dstImage", _dstImage);
	//GammaTransform(findline, findline);
	//cv::imshow("dstImage", findline);

	Mat drawline;
	int ksize1 = 3;
	int ksize2 = 3;
	double sigma1 = 10.0;
	double sigma2 = 20.0;
	GaussianBlur(dstImage, dstImage, cv::Size(ksize1, ksize2), sigma1, sigma2);
	//Mat tthreshold;
	//float otsu;
	//otsu = threshold(findline, tthreshold, 0, 255, THRESH_BINARY | THRESH_OTSU);
	//Scalar m = mean(findline);
	//
	cv::Mat gray_blur;
	cv::Mat element = cv::getStructuringElement(cv::MORPH_RECT, cv::Size(3, 3));
	cv::morphologyEx(dstImage, gray_blur, cv::MORPH_DILATE, element, cv::Point(-1, -1), 3);
	cv::Mat edges;

	std::vector<cv::Point> vertices;
	vertices.push_back(cv::Point(60, 0));
	vertices.push_back(cv::Point(-40, 170));
	vertices.push_back(cv::Point(340, 170));
	vertices.push_back(cv::Point(240, 0));
	Mat gray_mask;
	float me;
	getROI(gray_blur, vertices, gray_mask,me);
	pic_mean = threshold(gray_mask, gray_mask, 0, 255, THRESH_BINARY | THRESH_OTSU);
	
	cv::Canny(gray_blur, edges, 50, 180);
	//imshow("can", edges);
	cv::Mat masked;
	getROI(edges, vertices, masked,me);
	//imshow("masked", masked);
	std::vector<cv::Vec4i>lines;
	cv::HoughLinesP(masked, lines, 0.8, CV_PI / 45, 26,15, 50);
	cv::Mat hlines_img;
	cvtColor(findline, hlines_img, CV_GRAY2BGR);
	//cvtColor(findline, hlines_img, CV_GRAY2BGR);
	//drawLines(hlines_img, lines, cv::Scalar(255, 0, 0));
	//imshow("hough", hlines_img);
	//waitKey(0);
	
	filtered_lines.resize(2);
	//linesType filtered_lines;
	vector<line_rorect >left_lanes, right_lanes;
	//imwrite("pair.jpg", hlines_img);
	bypassangle_linewidth(lines,hough_left, hough_right, left_lanes, right_lanes);


	vector<line_rorect> white_left_lanes, white_right_lanes;
	drawLines_pair(hlines_img, left_lanes, Scalar(255, 0, 255));
	drawLines_pair(hlines_img, right_lanes, Scalar(255, 200, 0));
	//imshow("pair", hlines_img);
	if (left_lock&&left_c>0&& left_lanes.size()>left_c-1) {
		if (!save_left) {
			save_left = true;
			filtered_lines[0].state = true;
			filtered_lines[0].line_data = left_lanes[left_c - 1];
			jp6_camera._left_angle = filtered_lines[0].line_data.angle;
			jp6_camera._left_point_x = filtered_lines[0].line_data.x;
			jp6_camera._left_point_y = filtered_lines[0].line_data.y;
		}
	}
	if(!left_lock)
	{
		save_left = false;
		filtered_lines[0].state = false;
		//filtered_lines[0].state = false;
	}
	if (right_lock&&right_c>0&&right_lanes.size()>right_c - 1) {
		if (!save_right) {
			save_right = true;
			filtered_lines[1].state = true;
			filtered_lines[1].line_data = right_lanes[right_c - 1];
			jp6_camera._right_angle = filtered_lines[1].line_data.angle;
			jp6_camera._right_point_x = filtered_lines[1].line_data.x;
			jp6_camera._right_point_y = filtered_lines[1].line_data.y;
		}
	}
	if (!right_lock)
	{
		save_right = false;
		filtered_lines[1].state = false;
		//filtered_lines[0].state = false;
	}
	if (filtered_lines[0].state&&filtered_lines[1].state) {
		jp6_camera._double_lane_dis=abs(l2r_lane(filtered_lines[1].line_data, filtered_lines[0].line_data));
	}
	//bypass_color(gray_blur, left_lanes, white_left_lanes);
	//bypass_color(gray_blur, right_lanes, white_right_lanes);
	
	//bypassAngleFilter(gray_blur,lines, hough_left,hough_right,filtered_lines);
	//bypass_laneFilter(white_left_lanes, white_right_lanes, hough_left, hough_right, filtered_lines);
	//cv::Mat avg_img = cv::Mat::zeros(findline.size(), CV_8UC3);
	
	drawlrLines(hlines_img, filtered_lines);
	//refresh_lane(filtered_lines, hough_left, hough_right);
	imshow("lane", hlines_img);
	//imwrite("lane.jpg",hlines_img);
	waitKey(1);
	//waitKey(0);
	/*bool left_clean=false,right_clean=false;
	if (filtered_lines[1].state&&filtered_lines[0].state)
	{
		lane_start = true;		
		right_miss_num = 0;
		left_miss_num = 0;
	}
	if (filtered_lines[1].state)
	{
		right_miss_num = 0;
	}
	if (filtered_lines[0].state)
	{
		left_miss_num = 0;
	}
	if (right_miss_num >= _right_miss&&left_miss_num >= _left_miss)//miss lane
		lane_start = false;
	
	if (filtered_lines[1].state) {
		if (hough_right.angle < stand_right - change_angle)//line not found
		{
			//waring_turn = true;
			left_warning_num++;
		}
		else if (hough_right.angle > stand_right + change_angle / 1.3)
		{
			//waring_turn = true;
			right_warning_num++;
		}
		else {
			right_clean = true;
		}
	}
	if (filtered_lines[0].state) {
		if (hough_left.angle < stand_left - change_angle / 1.3)
		{
			//waring_turn = true;
			left_warning_num++;
		}
		else if (hough_left.angle > stand_left + change_angle)
		{
			//waring_turn = true;
			right_warning_num++;
		}
		else {
			left_clean = true;
		}
	}
	if (left_clean&&right_clean)
	{
		right_warning_num = 0;
		left_warning_num = 0;
		if(!in_lane)
			in_lane = true;
	}*/
	/*if (!in_lane)
	{
		turn_dir = 0;
		double lend = getTickCount();
		double nee=lend - start;
		totle_time+=nee;
		cout <<"now / average need "<<nee / cv::getTickFrequency() * 1000 <<" / "<< totle_time/find_line_num/cv::getTickFrequency() * 1000<< "ms." << endl;
		
		return turn_dir;
	}*/
	/*if (left_warning_num > _warn_num/2)
	{
		turn_dir = 1;
		putText(drawline, "L", Point(0, 20), 1, 2, Scalar(255, 0, 255), 2);
		cout << "LLLLLLLL  turn" << endl;
		left_miss_num = _left_miss;
		left_warning_num = 0;
		lane_init();
		in_lane = false;
		//waitKey();
	}
	else if (right_warning_num> _warn_num)
	{
		turn_dir = -1;
		putText(drawline, "R", Point(drawline.cols - 20, 20), 1, 2, Scalar(255, 0, 255), 2);
		cout << "RRRRRRRR  turn" << endl;
		right_miss_num = _right_miss;
		right_warning_num = 0;
		lane_init();
		in_lane = false;
		//waitKey();
	}
	//cout<<filtered_lines[0].state<<" "<<filtered_lines[1].state<<"  "<<lane_start<<endl;
	//waitKey(1);
	
	//else
	double lend = getTickCount();
	double nee=lend - start;
	totle_time+=nee;
	cout <<"now / average need "<<nee / cv::getTickFrequency() * 1000 <<" / "<< totle_time/find_line_num/cv::getTickFrequency() * 1000<< "ms." << endl;
	*/
	return turn_dir;
#if 0
	//
	//
	////waitKey(0);
	////cout << otsu << " line mean is " << m[0] << endl;
	//threshold(findline, findline, otsu - 10, 255, THRESH_BINARY|THRESH_OTSU );
	Mat mean1, std1;
	//findline = ~findline;
	meanStdDev(findline, mean1, std1);
	int th = mean1.at<double>(0) - std1.at<double>(0);
	int th2= mean1.at<double>(0)*1.1 + std1.at<double>(0)*0.8;

	cout << mean1.at<double>(0)<<"  "<< std1.at<double>(0) << endl;
	
	threshold(findline, findline, th, 255, THRESH_TOZERO);
	imshow("th1", findline);
	Mat ttt;
	float otsu=threshold(findline, ttt, 0, 255, THRESH_BINARY| THRESH_OTSU);
	cout << otsu << endl;
	threshold(findline, findline, th2, 255, THRESH_BINARY);

	cv::imshow("THRESH_BINARY", findline);
	waitKey(0);
	
	Mat dst(findline.rows + 20, findline.cols + 20, CV_8UC1);
	dst.setTo(0);
	findline.copyTo(dst(Rect(10, 10, findline.cols, findline.rows)));
	/*half_left.copyTo(dst(Rect(0, 0, half_left.cols, half_left.rows)));
	half_right.copyTo(dst(Rect(half_left.cols, 0, half_left.cols, half_left.rows)));*/
	//Mat element = getStructuringElement(0, Size(3, 3), Point(-1, -1));
	////erode(dst, dst, Mat(), Point(-1, -1),1, 0, 0);
	//dilate(findline, dst(Rect(10, 10, findline.cols, findline.rows)), element, Point(-1, -1),2);
	//resize(dst, dst, Size(dst.cols * 2, dst.rows * 2));
	morphologyEx(dst, dst, CV_MOP_OPEN, element);
	//resize(dst, dst, Size(dst.cols /2, dst.rows / 2));
	//imshow("b_dst", dst);

	vector<vector<Point> > contours;
	vector<Vec4i> hierarchy;

	findContours(dst, contours, hierarchy, CV_RETR_EXTERNAL, CV_CHAIN_APPROX_NONE);

	vector<RotatedRect> box(contours.size()); //������С��Ӿ��μ���
	float x, y, angle, height, width, change_angle = 10, left_angle = 0, xleft, yleft, xright, yright, leftxmin, rightxmin;
	int centercol = dst.cols / 2;
	int minleft = centercol, minright = centercol;
	bool waring_turn = true;//�������
	bool find_line = false;
	bool find_leftline = false, find_rightline = false, k_find_rightline = false, k_find_leftline = false;
	//imshow("thr", dst);
	cvtColor(dst, drawline, CV_GRAY2BGR);
	float leftcangle = line_left_standard, rightcangle = line_right_standard;
	Mat draw_rect;
	drawline.copyTo(draw_rect);


	/*****   kalman   ****/

	if (right_need_predict == 5)
	{
		predictionr = KFr.predict();

		//measurementr.at<float>(0) = rightcangle;

		right_predict_line.angle = predictionr.at<float>(0);
		right_predict_line.x = predictionr.at<float>(1);
		right_predict_line.y = predictionr.at<float>(2);

		//printf("pre %f  %f %f\n", right_predict_line.x, right_predict_line.y, right_predict_line.angle);
	}
	else if (right_need_predict < 0)//line not found
	{
		standright.copyTo(KFr.statePost);
		predictionr = KFr.predict();
		//measurementr.at<float>(0) = rightcangle;
		right_predict_line.angle = predictionr.at<float>(0);
		right_predict_line.x = predictionr.at<float>(1);
		right_predict_line.y = predictionr.at<float>(2);
		right_need_predict = 0;
	}

	if (left_need_predict == 5)
	{
		predictionl = KFl.predict();

		//measurementr.at<float>(0) = rightcangle;

		left_predict_line.angle = predictionl.at<float>(0);
		left_predict_line.x = predictionl.at<float>(1);
		left_predict_line.y = predictionl.at<float>(2);

		//printf("pre %f  %f %f\n", right_predict_line.x, right_predict_line.y, right_predict_line.angle);
	}
	else if (left_need_predict < 0)//line not found
	{
		standleft.copyTo(KFl.statePost);
		predictionl = KFl.predict();

		//measurementr.at<float>(0) = rightcangle;

		left_predict_line.angle = predictionl.at<float>(0);
		left_predict_line.x = predictionl.at<float>(1);
		left_predict_line.y = predictionl.at<float>(2);
		left_need_predict = 0;
	}
	line_rorect cur_line;
	


	for (int i = 0; i < contours.size(); i++)
	{

		box[i] = minAreaRect(Mat(contours[i]));  //����ÿ��������С��Ӿ���
		height = box[i].size.height;
		width = box[i].size.width;
		angle = box[i].angle;
		x = box[i].center.x;
		y = box[i].center.y;
		if (min(height, width) < 4)
			continue;
		Point2f vertices[4];

		box[i].points(vertices);

		char name[10];
		sprintf_s(name, "%d", i);
		double area = contourArea(contours[i]);
		double precent = area / (max(height, width) * min(height, width)) * 100;//���ռ��
																				//printf("i:%d----->area��%3.2f�� x:%3.2f, y:%3.2f, w:%3.2f, h:%3.2f, a:%3.2f\n", i, precent, x, y, width, height, angle);
																				//imshow("box", draw_rect);

		float box_scal = max(height, width) / (height + width);

		if (box_scal < 0.7)
			continue;
		Vec4f line_data, _fitline;
		Vec2f line_data_double;
		float r2 = 0;
		fitLine(contours[i], _fitline, DIST_L12, 0, 1e-2, 1e-2);
		//min2line(contours[i], line_data);
		float max_r2 = 0;
		vector<min2line_Struct> max_r2_and_state(2);
		vector<min2line_Struct> r2_and_state(2);


		line_data[0] = _fitline[2];
		line_data[1] = _fitline[3];
		line_data[2] = _fitline[1] / _fitline[0];
		line_data[3] = -line_data[2] * line_data[0] + line_data[1];
		fit_line(draw_rect, contours[i], box_scal, line_data, r2_and_state);
		if (!draw2line(drawline, r2_and_state, area, height, width, cur_line))
			continue;

		if (fabs(right_predict_line.angle - cur_line.angle) < change_angle)
		{
			float angle_diff = l2l_angle(cur_line, right_predict_line);
			if (angle_diff < minangle_r)
			{
				true_right_line.x = cur_line.x;
				true_right_line.y = cur_line.y;
				true_right_line.angle = cur_line.angle;
				minangle_r = angle_diff;
				k_find_rightline = true;
			}
			//}
		}
		if (fabs(left_predict_line.angle - cur_line.angle) < change_angle)
		{
			float angle_diff = l2l_angle(cur_line, left_predict_line);
			if (angle_diff < minangle_l)
			{
				true_left_line.x = cur_line.x;
				true_left_line.y = cur_line.y;
				true_left_line.angle = cur_line.angle;
				minangle_l = angle_diff;
				k_find_leftline = true;
			}
			//}
		}
	}//for i end
	 /**************** found right line *********************/
	if (k_find_rightline) {
		measurementr.at<float>(0) = true_right_line.angle;
		measurementr.at<float>(1) = true_right_line.x;
		measurementr.at<float>(2) = true_right_line.y;
		KFr.correct(measurementr);// refresh kalman
		right_need_predict = 5;

	}
	else {
		right_need_predict--;
	}
	if (right_need_predict > 0)
	{
		int line_x_up = true_right_line.y - tan(true_right_line.angle / 180 * PI) * true_right_line.x;
		int line_x_down = true_right_line.y + tan(true_right_line.angle / 180 * PI) * (drawline.cols - true_right_line.x);
		line(drawline, Point(0, line_x_up), Point(drawline.cols, line_x_down), Scalar(255, 128, 128), 2);

	}
	/**************** found left line *********************/
	if (k_find_leftline) {
		measurementl.at<float>(0) = true_left_line.angle;
		measurementl.at<float>(1) = true_left_line.x;
		measurementl.at<float>(2) = true_left_line.y;
		KFl.correct(measurementl);// refresh kalman
		left_need_predict = 5;
	}
	else {
		left_need_predict--;
	}
	if (left_need_predict > 0)
	{
		int line_x_up = true_left_line.y - tan(true_left_line.angle / 180 * PI) * true_left_line.x;
		int line_x_down = true_left_line.y + tan(true_left_line.angle / 180 * PI) * (drawline.cols - true_left_line.x);
		line(drawline, Point(0, line_x_up), Point(drawline.cols, line_x_down), Scalar(0, 255, 128), 2);

	}
	bool right_toclean=true,left_toclean=true;
	if (k_find_rightline)
	{
		right_toclean = false;
		if (true_right_line.angle < stand_right - change_angle)//line not found
		{
			//waring_turn = true;
			left_warning_num++;
		}
		else if (true_right_line.angle > stand_right + change_angle)
		{
			//waring_turn = true;
			right_warning_num++;
		}
		else
		{
			right_toclean = true;

		}
	}
	
	if (k_find_leftline)//line  found
	{
		left_toclean = false;
		if (true_left_line.angle < stand_left - change_angle)
		{
			//waring_turn = true;
			left_warning_num++;
		}
		else if (true_left_line.angle > stand_left + change_angle)
		{
			//waring_turn = true;
			right_warning_num++;
		}
		else
		{
			left_toclean = true;

		}
	}
	if (left_toclean&&right_toclean)
	{
		right_warning_num = 0;
		left_warning_num = 0;
	}
	if (left_warning_num > _warn_num)
	{
		putText(drawline, "L", Point(0, 20), 1, 2, Scalar(255, 0, 255), 2);
		cout << "LLLLLLLL  turn" << endl;
		standright.copyTo(KFr.statePost);
		predictionr = KFr.predict();
		right_predict_line.angle = predictionr.at<float>(0);
		right_predict_line.x = predictionr.at<float>(1);
		right_predict_line.y = predictionr.at<float>(2);
		standleft.copyTo(KFl.statePost);
		predictionl = KFl.predict();
		left_predict_line.angle = predictionl.at<float>(0);
		left_predict_line.x = predictionl.at<float>(1);
		left_predict_line.y = predictionl.at<float>(2);
		right_need_predict = 0;
		left_warning_num = 0;
	}
	else if (right_warning_num> _warn_num)
	{
		//turn_dir = -1;
		putText(drawline, "R", Point(drawline.cols - 20, 20), 1, 2, Scalar(255, 0, 255), 2);
		cout << "RRRRRRRR  turn" << endl;
		standright.copyTo(KFr.statePost);
		predictionr = KFr.predict();
		right_predict_line.angle = predictionr.at<float>(0);
		right_predict_line.x = predictionr.at<float>(1);
		right_predict_line.y = predictionr.at<float>(2);

		standleft.copyTo(KFl.statePost);
		predictionl = KFl.predict();
		left_predict_line.angle = predictionl.at<float>(0);
		left_predict_line.x = predictionl.at<float>(1);
		left_predict_line.y = predictionl.at<float>(2);
		right_need_predict = 0;
		right_warning_num = 0;
	}
#endif
	//cvNamedWindow("line1", 1);


	return turn_dir;
}
