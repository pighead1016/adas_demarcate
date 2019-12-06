#include "stdafx.h"
#include "mars_smokephone.h"
//#include "include/net.h"

#include <opencv2/opencv.hpp>

#include <vector>

using namespace std;
using std::string;
/* Pair (label, confidence) representing a prediction. */
typedef std::pair<string, float> Prediction;
ncnn::Net nose_arm_mark;

// Use op::round/max/min for basic types (int, char, long, float, double, etc). Never with classes! std:: alternatives uses 'const T&' instead of 'const T' as argument.
// Round functions
// Signed
template<typename T>
inline char charRound(const T a)
{
	return char(a + 0.5f);
}

template<typename T>
inline signed char sCharRound(const T a)
{
	return (signed char)(a + 0.5f);
}

template<typename T>
inline int intRound(const T a)
{
	return int(a + 0.5f);
}

template<typename T>
inline long longRound(const T a)
{
	return long(a + 0.5f);
}

template<typename T>
inline long long longLongRound(const T a)
{
	return (long long)(a + 0.5f);
}

// Unsigned
template<typename T>
inline unsigned char uCharRound(const T a)
{
	return (unsigned char)(a + 0.5f);
}

template<typename T>
inline unsigned int uIntRound(const T a)
{
	return (unsigned int)(a + 0.5f);
}

template<typename T>
inline unsigned long ulongRound(const T a)
{
	return (unsigned long)(a + 0.5f);
}

template<typename T>
inline unsigned long long uLongLongRound(const T a)
{
	return (unsigned long long)(a + 0.5f);
}

// Max/min functions
template<typename T>
inline T fastMax(const T a, const T b)
{
	return (a > b ? a : b);
}

template<typename T>
inline T fastMin(const T a, const T b)
{
	return (a < b ? a : b);
}

template<class T>
inline T fastTruncate(T value, T min = 0, T max = 1)
{
	return fastMin(max, fastMax(min, value));
}

struct BlobData {
	int count;
	float* list;
	int num;
	int channels;
	int height;
	int width;
	int capacity_count;		//����ռ��Ԫ�ظ������ȣ��ֽ����� * sizeof(float)
};
//const std::vector<unsigned int> TIRED_RENDER{ 0,2,5,17 };
const std::vector<unsigned int> POSE_COCO_PAIRS_RENDER{ 1 , 8,1, 2 , 1, 5,
2, 3, 3, 4, 5, 6, 6, 7,
8, 9, 8,12,
9,10,10,11,
12,13,13,14,
1, 0, 0,16, 0,15,15,17,16,18,
2,17, 5,18 };

const unsigned int POSE_MAX_PEOPLE = 5;
const std::vector<unsigned int> TIRED_RENDER{ 0,2,5};
const std::vector<unsigned int> FACE_RENDER{ 0,15,16,17,18 };
const std::vector<unsigned int> EYESHOUDER_RENDER{ 15,16,2,5,1 };
//656x368
cv::Mat getImage(const cv::Mat& im, cv::Size baseSize = cv::Size(656, 368), float* scale = 0) {
	int w = baseSize.width;
	int h = baseSize.height;
	int nh = h;
	float s = h / (float)im.rows;;
	int nw = im.cols * s;

	if (nw > w) {
		nw = w;
		s = w / (float)im.cols;
		nh = im.rows * s;
	}

	if (scale)*scale = 1 / s;
	cv::Rect dst(0, 0, nw, nh);
	cv::Mat bck = cv::Mat::zeros(h, w, CV_8UC3);
	cv::resize(im, bck(dst), cv::Size(nw, nh));
	return bck;
}

//���ݵõ��Ľ����������������
void connectBodyPartsCpu(vector<float>& poseKeypoints, const float* const heatMapPtr, const float* const peaksPtr,
	const cv::Size& heatMapSize, const int maxPeaks, const int interMinAboveThreshold,
	const float interThreshold, const int minSubsetCnt, const float minSubsetScore, const float scaleFactor, vector<int>& keypointShape)
{
	keypointShape.resize(3);
	/*******************body 19************************
	const std::vector<unsigned int> POSE_COCO_PAIRS{  1, 5 , 1, 2,
		2, 3, 3, 4, 5, 6, 6, 7,
		1 , 8, 8, 9, 9,10,
		1, 11,11,12,12,13,
		
		1, 0, 0,14, 14,16,0,15,15,17,
		2,16, 5,17
		//14,19,19,20,14,21,
		//11,22,22,23,11,24
	};
	const std::vector<unsigned int> POSE_COCO_MAP_IDX{ 39,40,31,32,
		33,34,35,36,41,42,43,44,
		19,20,21,22,23,24,
		25,26,27,28,29,30,

		47,48,49,50,53,54,51,52,55,56,
		37,38,45,46
		//66,67,68,69,70,71,
		//72,73,74,75,76,77
	};/*
	/*******************body 25************************/
	const std::vector<unsigned int> POSE_COCO_PAIRS{ 1 , 8,1, 2 , 1, 5,
		2, 3, 3, 4, 5, 6, 6, 7,
		8, 9, 8,12,
		9,10,10,11,
		12,13,13,14,
		1, 0, 0,16, 0,15,15,17,16,18,
		2,17, 5,18
		//14,19,19,20,14,21,
		//11,22,22,23,11,24
	};
	const std::vector<unsigned int> POSE_COCO_MAP_IDX{ 26,27,40,41,48,49,
		42,43,44,45,50,51,52,53,
		32,33,34,35,
		28,29,30,31,
		36,37,38,39,
		56,57,60,61,58,59,62,63,64,65,
		46,47,54,55
		//66,67,68,69,70,71,
		//72,73,74,75,76,77
	};
	
	/****************************************************/
	const auto& bodyPartPairs = POSE_COCO_PAIRS;
	const auto& mapIdx = POSE_COCO_MAP_IDX;
	const auto numberBodyParts = 25;//change from 25

	const auto numberBodyPartPairs = bodyPartPairs.size() / 2;

	std::vector<std::pair<std::vector<int>, double>> subset;    // Vector<int> = Each body part + body parts counter; double = subsetScore
	const auto subsetCounterIndex = numberBodyParts;
	const auto subsetSize = numberBodyParts + 1;

	const auto peaksOffset = 3 * (maxPeaks + 1);
	const auto heatMapOffset = heatMapSize.area();

	for (auto pairIndex = 0u; pairIndex < numberBodyPartPairs; pairIndex++)
	{
		const auto bodyPartA = bodyPartPairs[2 * pairIndex];
		const auto bodyPartB = bodyPartPairs[2 * pairIndex + 1];
		const auto* candidateA = peaksPtr + bodyPartA*peaksOffset;
		const auto* candidateB = peaksPtr + bodyPartB*peaksOffset;
		const auto nA = intRound(candidateA[0]);
		const auto nB = intRound(candidateB[0]);

		// add parts into the subset in special case
		if (nA == 0 || nB == 0)
		{
			// Change w.r.t. other
			if (nA == 0) // nB == 0 or not
			{
				for (auto i = 1; i <= nB; i++)
				{
					bool num = false;
					const auto indexB = bodyPartB;
					for (auto j = 0u; j < subset.size(); j++)
					{
						const auto off = (int)bodyPartB*peaksOffset + i * 3 + 2;
						if (subset[j].first[indexB] == off)
						{
							num = true;
							break;
						}
					}
					if (!num)
					{
						std::vector<int> rowVector(subsetSize, 0);
						rowVector[bodyPartB] = bodyPartB*peaksOffset + i * 3 + 2; //store the index
						rowVector[subsetCounterIndex] = 1; //last number in each row is the parts number of that person
						const auto subsetScore = candidateB[i * 3 + 2]; //second last number in each row is the total score
						subset.emplace_back(std::make_pair(rowVector, subsetScore));
					}
				}
			}
			else // if (nA != 0 && nB == 0)
			{
				for (auto i = 1; i <= nA; i++)
				{
					bool num = false;
					const auto indexA = bodyPartA;
					for (auto j = 0u; j < subset.size(); j++)
					{
						const auto off = (int)bodyPartA*peaksOffset + i * 3 + 2;
						if (subset[j].first[indexA] == off)
						{
							num = true;
							break;
						}
					}
					if (!num)
					{
						std::vector<int> rowVector(subsetSize, 0);
						rowVector[bodyPartA] = bodyPartA*peaksOffset + i * 3 + 2; //store the index
						rowVector[subsetCounterIndex] = 1; //last number in each row is the parts number of that person
						const auto subsetScore = candidateA[i * 3 + 2]; //second last number in each row is the total score
						subset.emplace_back(std::make_pair(rowVector, subsetScore));
					}
				}
			}
		}
		else // if (nA != 0 && nB != 0)
		{
			std::vector<std::tuple<double, int, int>> temp;
			const auto numInter = 10;
			const auto* const mapX = heatMapPtr + mapIdx[2 * pairIndex] * heatMapOffset;
			const auto* const mapY = heatMapPtr + mapIdx[2 * pairIndex + 1] * heatMapOffset;
			for (auto i = 1; i <= nA; i++)
			{
				for (auto j = 1; j <= nB; j++)
				{
					const auto dX = candidateB[j * 3] - candidateA[i * 3];
					const auto dY = candidateB[j * 3 + 1] - candidateA[i * 3 + 1];
					const auto normVec = float(std::sqrt(dX*dX + dY*dY));
					// If the peaksPtr are coincident. Don't connect them.
					if (normVec > 1e-6)
					{
						const auto sX = candidateA[i * 3];
						const auto sY = candidateA[i * 3 + 1];
						const auto vecX = dX / normVec;
						const auto vecY = dY / normVec;

						auto sum = 0.;
						auto count = 0;
						for (auto lm = 0; lm < numInter; lm++)
						{
							const auto mX = fastMin(heatMapSize.width - 1, intRound(sX + lm*dX / numInter));
							const auto mY = fastMin(heatMapSize.height - 1, intRound(sY + lm*dY / numInter));
							//checkGE(mX, 0, "", __LINE__, __FUNCTION__, __FILE__);
							//checkGE(mY, 0, "", __LINE__, __FUNCTION__, __FILE__);
							const auto idx = mY * heatMapSize.width + mX;
							const auto score = (vecX*mapX[idx] + vecY*mapY[idx]);
							if (score > interThreshold)
							{
								sum += score;
								count++;
							}
						}
						// parts score + connection score
						if (count > interMinAboveThreshold)
							temp.emplace_back(std::make_tuple(sum / count, i, j));
					}
				}
			}

			// select the top minAB connection, assuming that each part occur only once
			// sort rows in descending order based on parts + connection score
			if (!temp.empty())
				std::sort(temp.begin(), temp.end(), std::greater<std::tuple<float, int, int>>());

			std::vector<std::tuple<int, int, double>> connectionK;

			const auto minAB = fastMin(nA, nB);
			std::vector<int> occurA(nA, 0);
			std::vector<int> occurB(nB, 0);
			auto counter = 0;
			for (auto row = 0u; row < temp.size(); row++)
			{
				const auto score = std::get<0>(temp[row]);
				const auto x = std::get<1>(temp[row]);
				const auto y = std::get<2>(temp[row]);
				if (!occurA[x - 1] && !occurB[y - 1])
				{
					connectionK.emplace_back(std::make_tuple(bodyPartA*peaksOffset + x * 3 + 2,
						bodyPartB*peaksOffset + y * 3 + 2,
						score));
					counter++;
					if (counter == minAB)
						break;
					occurA[x - 1] = 1;
					occurB[y - 1] = 1;
				}
			}
			// Cluster all the body part candidates into subset based on the part connection
			// initialize first body part connection 15&16
			if (pairIndex == 0)
			{
				for (const auto connectionKI : connectionK)
				{
					std::vector<int> rowVector(numberBodyParts + 3, 0);
					const auto indexA = std::get<0>(connectionKI);
					const auto indexB = std::get<1>(connectionKI);
					const auto score = std::get<2>(connectionKI);
					rowVector[bodyPartPairs[0]] = indexA;
					rowVector[bodyPartPairs[1]] = indexB;
					rowVector[subsetCounterIndex] = 2;
					// add the score of parts and the connection
					const auto subsetScore = peaksPtr[indexA] + peaksPtr[indexB] + score;
					subset.emplace_back(std::make_pair(rowVector, subsetScore));
				}
			}
			// Add ears connections (in case person is looking to opposite direction to camera)
			else if (pairIndex == 17 || pairIndex == 16)
			{
				for (const auto& connectionKI : connectionK)
				{
					const auto indexA = std::get<0>(connectionKI);
					const auto indexB = std::get<1>(connectionKI);
					for (auto& subsetJ : subset)
					{
						auto& subsetJFirst = subsetJ.first[bodyPartA];
						auto& subsetJFirstPlus1 = subsetJ.first[bodyPartB];
						if (subsetJFirst == indexA && subsetJFirstPlus1 == 0)
							subsetJFirstPlus1 = indexB;
						else if (subsetJFirstPlus1 == indexB && subsetJFirst == 0)
							subsetJFirst = indexA;
					}
				}
			}
			else
			{
				if (!connectionK.empty())
				{
					// A is already in the subset, find its connection B
					for (auto i = 0u; i < connectionK.size(); i++)
					{
						const auto indexA = std::get<0>(connectionK[i]);
						const auto indexB = std::get<1>(connectionK[i]);
						const auto score = std::get<2>(connectionK[i]);
						auto num = 0;
						for (auto j = 0u; j < subset.size(); j++)
						{
							if (subset[j].first[bodyPartA] == indexA)
							{
								subset[j].first[bodyPartB] = indexB;
								num++;
								subset[j].first[subsetCounterIndex] = subset[j].first[subsetCounterIndex] + 1;
								subset[j].second = subset[j].second + peaksPtr[indexB] + score;
							}
						}
						// if can not find partA in the subset, create a new subset
						if (num == 0)
						{
							std::vector<int> rowVector(subsetSize, 0);
							rowVector[bodyPartA] = indexA;
							rowVector[bodyPartB] = indexB;
							rowVector[subsetCounterIndex] = 2;
							const auto subsetScore = peaksPtr[indexA] + peaksPtr[indexB] + score;
							subset.emplace_back(std::make_pair(rowVector, subsetScore));
						}
					}
				}
			}
		}
	}

	// Delete people below the following thresholds:
	// a) minSubsetCnt: removed if less than minSubsetCnt body parts
	// b) minSubsetScore: removed if global score smaller than this
	// c) POSE_MAX_PEOPLE: keep first POSE_MAX_PEOPLE people above thresholds
	auto numberPeople = 0;
	std::vector<int> validSubsetIndexes;
	validSubsetIndexes.reserve(fastMin((size_t)POSE_MAX_PEOPLE, subset.size()));
	for (auto index = 0u; index < subset.size(); index++)
	{
		const auto subsetCounter = subset[index].first[subsetCounterIndex];
		const auto subsetScore = subset[index].second;
		if (subsetCounter >= minSubsetCnt && (subsetScore / subsetCounter) > minSubsetScore)
		{
			numberPeople++;
			validSubsetIndexes.emplace_back(index);
			if (numberPeople == POSE_MAX_PEOPLE)
				break;
		}
		else if (subsetCounter < 1)
			printf("Bad subsetCounter. Bug in this function if this happens. %d, %s, %s", __LINE__, __FUNCTION__, __FILE__);
	}

	// Fill and return poseKeypoints
	keypointShape = { numberPeople, (int)numberBodyParts, 3 };
	if (numberPeople > 0)
		poseKeypoints.resize(numberPeople * (int)numberBodyParts * 3);
	else
		poseKeypoints.clear();

	for (auto person = 0u; person < validSubsetIndexes.size(); person++)
	{
		const auto& subsetI = subset[validSubsetIndexes[person]].first;
		for (auto bodyPart = 0u; bodyPart < numberBodyParts; bodyPart++)
		{
			const auto baseOffset = (person*numberBodyParts + bodyPart) * 3;
			const auto bodyPartIndex = subsetI[bodyPart];
			if (bodyPartIndex > 0)
			{
				poseKeypoints[baseOffset] = peaksPtr[bodyPartIndex - 2] * scaleFactor;
				poseKeypoints[baseOffset + 1] = peaksPtr[bodyPartIndex - 1] * scaleFactor;
				poseKeypoints[baseOffset + 2] = peaksPtr[bodyPartIndex];
			}
			else
			{
				poseKeypoints[baseOffset] = 0.f;
				poseKeypoints[baseOffset + 1] = 0.f;
				poseKeypoints[baseOffset + 2] = 0.f;
			}
		}
	}
}
float cos_vector(cv::Point2f p1, cv::Point2f p2, cv::Point2f p3, cv::Point2f p4)
{
	float v1x = p1.x - p2.x;
	float v1y = p1.y - p2.y;
	float v2x = p3.x - p4.x;
	float v2y = p3.y - p4.y;
	return (v1x*v2x + v1y*v2y) / sqrt((v1x*v1x + v1y*v1y)*(v2x*v2x + v2y*v2y));
}
//topShape[1] = bottomShape[1] - 1; // Number parts + bck - 1      77 = 78 - 1
//topShape[2] = maxPeaks + 1; // # maxPeaks + 1                    5 = 5 + 1
//topShape[3] = 3;  // X, Y, score                                 3

//bottom_blob�����룬top�����
void nms(BlobData* bottom_blob, BlobData* top_blob, float threshold) {
	int w = bottom_blob->width;
	int h = bottom_blob->height;
	int plane_offset = w * h;
	float* ptr = bottom_blob->list;
	float* top_ptr = top_blob->list;
	int top_plane_offset = top_blob->width * top_blob->height;
	int max_peaks = top_blob->height - 1;

	for (int n = 0; n < bottom_blob->num; ++n) {
		for (int c = 0; c < bottom_blob->channels - 1; ++c) {

			int num_peaks = 0;
			for (int y = 1; y < h - 1 && num_peaks != max_peaks; ++y) {
				for (int x = 1; x < w - 1 && num_peaks != max_peaks; ++x) {
					float value = ptr[y*w + x];
					if (value > threshold) {
						const float topLeft = ptr[(y - 1)*w + x - 1];
						const float top = ptr[(y - 1)*w + x];
						const float topRight = ptr[(y - 1)*w + x + 1];
						const float left = ptr[y*w + x - 1];
						const float right = ptr[y*w + x + 1];
						const float bottomLeft = ptr[(y + 1)*w + x - 1];
						const float bottom = ptr[(y + 1)*w + x];
						const float bottomRight = ptr[(y + 1)*w + x + 1];

						if (value > topLeft && value > top && value > topRight
							&& value > left && value > right
							&& value > bottomLeft && value > bottom && value > bottomRight)
						{
							//��������������
							float xAcc = 0;
							float yAcc = 0;
							float scoreAcc = 0;
							for (int kx = -3; kx <= 3; ++kx) {
								int ux = x + kx;
								if (ux >= 0 && ux < w) {
									for (int ky = -3; ky <= 3; ++ky) {
										int uy = y + ky;
										if (uy >= 0 && uy < h) {
											float score = ptr[uy * w + ux];
											xAcc += ux * score;
											yAcc += uy * score;
											scoreAcc += score;
										}
									}
								}
							}

							xAcc /= scoreAcc;
							yAcc /= scoreAcc;
							scoreAcc = value;
							top_ptr[(num_peaks + 1) * 3 + 0] = xAcc;
							top_ptr[(num_peaks + 1) * 3 + 1] = yAcc;
							top_ptr[(num_peaks + 1) * 3 + 2] = scoreAcc;
							num_peaks++;
						}
					}
				}
			}
			top_ptr[0] = num_peaks;
			ptr += plane_offset;
			top_ptr += top_plane_offset;
		}
	}
}
float p2p_distance(cv::Point2f p1, cv::Point2f p2)
{
	return (p1.x - p2.x)*(p1.x - p2.x) + (p1.y - p2.y)*(p1.y - p2.y);
}
#define M_PI       3.14159265358979323846

float p2p_angle(cv::Point2f org, cv::Point2f p)
{
	return atan2f(org.y - p.y, p.x - org.x);
}

bool renderKeypointsCpu(cv::Mat& frame, const vector<float>& keypoints, vector<int> keyshape,const float threshold, float scale,cv::Point2f & nose_p_from_keypoint_temp,float& rear_angle,float& turn_face,float& turn_ear,int left_point_detection)
{
	bool ret = false;
	//smoke_phone_action_temp = false;
	// Get frame channels
	const auto numberKeypoints = keyshape[1];
	//cv::Mat oo = cv::Mat::zeros(frame.rows * 2, frame.cols, CV_8UC3);
	cv::Mat showcolor=frame.clone();
	/////draw line
	
	
	// Keypoints
	for (auto person = 0; person < keyshape[0]; person++)
	{
		cv::Point2f tired_p[4];
		bool tired_flag = true;
		for (size_t tie_num = 0; tie_num < TIRED_RENDER.size(); tie_num++)
		{
			int tired_part= (person * numberKeypoints + TIRED_RENDER[tie_num]) * keyshape[2];
			tired_flag = tired_flag & (keypoints[tired_part + 2] > threshold);
			tired_p[tie_num] = cv::Point2f(keypoints[tired_part], keypoints[tired_part + 1]);
		}
		if (tired_flag)
		{
		//	float nose_at
			turn_face = (tired_p[0].x - tired_p[1].x) /(tired_p[2].x - tired_p[1].x);
			
		}
			int tired_part= (person * numberKeypoints + left_point_detection) * keyshape[2];
			tired_flag = tired_flag & (keypoints[tired_part + 2] > threshold);
			tired_p[3] = cv::Point2f(keypoints[tired_part], keypoints[tired_part + 1]);
		if (tired_flag)
		{
			turn_ear = (tired_p[3].x - tired_p[1].x) / (tired_p[2].x - tired_p[1].x);
			float shouder_angle = p2p_angle(tired_p[1], tired_p[2]);
	//		float nose_angle = p2p_angle(tired_p[2], tired_p[0]);
			float _rear_angle= p2p_angle(tired_p[0],tired_p[3]);
		//	float nose_dis = sin(nose_angle - shouder_angle)*p2p_distance(tired_p[0], tired_p[2]) / p2p_distance(tired_p[1], tired_p[2]);
			rear_angle = _rear_angle - shouder_angle;
		}
		// Size-dependent variables
		bool left_arm = false, right_arm = false,nose=false,lshouder=false,rshouder=false;
		vector<cv::Point2f> arm_p(7, cv::Point(-1.0, -1.0));
		// Draw circles
		cv::Point2f nose_p;
		for (auto part = 0; part < 8; part++)
		{
			const auto faceIndex = (person * numberKeypoints + part) * keyshape[2];
			bool flag = keypoints[faceIndex + 2] > threshold;
			if (part == 0)
			{
				nose = flag;
				if (nose)
				{
					left_arm = true;
					right_arm = true;
					nose_p= { keypoints[faceIndex] * scale ,keypoints[faceIndex + 1] * scale };
					//ret = true;
					nose_p_from_keypoint_temp=nose_p;
				}
				else
					break;// Not found nose
			}
			if (part<8 && part>0)
			{
				if(flag)
				{
					arm_p[part - 1] = { keypoints[faceIndex] * scale ,keypoints[faceIndex + 1] * scale };
					//cv::circle(oo,arm_p[part-1],2,Scalar(0,255,0),2,1);
					cv::circle(frame, arm_p[part - 1], 2, cv::Scalar(0, 255, 0), 2, 1);
				}
				
				if (part > 4)//right_arm 567
					right_arm = right_arm&flag;
				else if (part > 1)//left_arm 234
					left_arm = left_arm&flag;
				else//neck 1
				{
					left_arm = true;
					right_arm = true;
				}
			}
			if (part == 2)
				lshouder = flag;
			if (part == 5)
				rshouder = flag;
		}
		//std::cout << person << "/" << keyshape[0] << ":" << left_arm << right_arm << std::endl;
		if (right_arm&&nose)
		{
			float angle2;
			bool _near, hup = false, belt = false;// = (p2p_distance(nose_p, arm_p[6])<p2p_distance(arm_p[5], arm_p[6]));//nose is nearer the wrist than elbow
			cv::line(showcolor, arm_p[4], arm_p[5], cv::Scalar(255, 0, 255), 3, 4);
			cv::line(showcolor, arm_p[6], arm_p[5], cv::Scalar(255, 255, 0), 3, 4);
			angle2 = cos_vector(arm_p[5], arm_p[4], arm_p[6], arm_p[5]);
			float rhx = (3 * arm_p[6].x - arm_p[5].x) / 2;
			float rhy = (3 * arm_p[6].y - arm_p[5].y) / 2;
			if (lshouder)
			{
				//cout<<arm_p[1]<<" 4 "<<arm_p[4]<<" 5 "<<arm_p[5]<<" 6 "<<arm_p[6]<<" "<<rhx<< "  "<<rhy<<endl;
				float a1 = atan2f(arm_p[1].y - arm_p[4].y, arm_p[4].x - arm_p[1].x);
				float a2 = atan2f(arm_p[1].y - rhy, rhx - arm_p[1].x);
				float right_hand_up = a2 - a1;
				if (right_hand_up > M_PI)
					right_hand_up -= 2 * M_PI;
				if (right_hand_up < -M_PI)
					right_hand_up += 2 * M_PI;
				if (right_hand_up > 0 && right_hand_up < M_PI)
					hup = true;
				//cout << "right:angle,distance-> " <<  angle2 << "," << a1<<":"<<a2 << endl;
			}
			if (angle2 < 0)
			{
				cv::putText(showcolor, "R", cv::Point(frame.cols - 20, 20), cv::FONT_HERSHEY_PLAIN, 2, cv::Scalar(0, 0, 255), 2);
				//cv::imwrite("R.jpg", showcolor);	
			}
			float belt_socer = cos_vector(arm_p[1], arm_p[5], arm_p[6], arm_p[5]);
			if (fabs(belt_socer) < 0.99)
			{
				belt = true;
				//cv::waitKey(0);
			}
			_near=true;//ignore nose distance
			if (angle2<-0.3 && hup&&belt)
			{
				//cv::putText(frame, "R", cv::Point(frame.cols - 20, 20), cv::FONT_HERSHEY_PLAIN, 2, cv::Scalar(0, 0, 255), 2);
				//smoke_phone_action_temp = true;
				nose_p_from_keypoint_temp=nose_p;
				//cv::imwrite("R.jpg", frame);
				return true;
			}
		}
		if (left_arm&&nose)
		{
			float angle2;
			bool _near,hup=false;// = (p2p_distance(nose_p, arm_p[3])<p2p_distance(arm_p[2], arm_p[3]));//nose is nearer the wrist than elbow
			cv::line(showcolor,arm_p[2],arm_p[1],cv::Scalar(0, 0, 255), 3, 4);
			cv::line(showcolor,arm_p[2],arm_p[3],cv::Scalar(0, 255, 255), 3, 4);
			angle2 = cos_vector(arm_p[2], arm_p[1], arm_p[3], arm_p[2]);
			float lhx=(3*arm_p[3].x-arm_p[2].x)/2;
			float lhy=(3*arm_p[3].y-arm_p[2].y)/2;
			if(rshouder)
			{
				
				float a1=atan2f(arm_p[4].y-arm_p[1].y,arm_p[1].x-arm_p[4].x);
				float a2=atan2f(arm_p[4].y-lhy,lhx-arm_p[4].x);
				float hand_up=a2-a1;
				if (hand_up>M_PI)
					hand_up-=2*M_PI;
				if (hand_up<-M_PI)
					hand_up+=2*M_PI;
				if(hand_up<0&&hand_up>-M_PI)
					hup=true;
				//cout << "left:angle,distance-> " <<  angle2 << "," << a1<<":"<<a2 << endl;
			}			
			if(angle2<0)
			{
				
				cv::putText(showcolor, "L", cv::Point(0, 20), cv::FONT_HERSHEY_PLAIN, 2, cv::Scalar(0, 0, 255), 2);
				//cv::imwrite("L.jpg", showcolor);	
			}
			//cout << "left:diraction,angle,distance-> " << angle2 << "," << near << endl;
			
			_near=true;
			if (angle2 < -0.5 && hup)
			{
				//cv::putText(frame, "L", cv::Point(0, 20), cv::FONT_HERSHEY_PLAIN, 2, cv::Scalar(0, 0, 255), 2);
				//smoke_phone_action_temp = true;
				nose_p_from_keypoint_temp=nose_p;
				//cv::imwrite("L.jpg", frame);
				return true;
			}
		}
	}
	
	
	return ret;
}
BlobData* createBlob_local(int num, int channels, int height, int width) {
	BlobData* blob = new BlobData();
	blob->num = num;
	blob->width = width;
	blob->channels = channels;
	blob->height = height;
	blob->count = num*width*channels*height;
	blob->list = new float[blob->count];
	blob->capacity_count = blob->count;
	return blob;
}

BlobData* createEmptyBlobData() {
	BlobData* blob = new BlobData();
	memset(blob, 0, sizeof(*blob));
	return blob;
}

void releaseBlob_local(BlobData** blob) {
	if (blob) {
		BlobData* ptr = *blob;
		if (ptr) {
			if (ptr->list)
				delete[] ptr->list;

			delete ptr;
		}
		*blob = 0;
	}
}

bool key_point(cv::Mat image_gray,cv::Point2f& nose_p_from_keypoint_temp,float& rear_angle, float& turn_face,float& turn_ear,int & act_peonum_temp,int left_point_detection)
{
	//image_gray = cv::imread("d://workspace/ll20191205154506.jpg",0);
	//image_gray = image_gray(cv::Rect(0, 0, image_gray.cols , image_gray.cols));
	cv::Mat image;
	cv::cvtColor(image_gray,image,CV_GRAY2BGR);
	//image = image(cv::Rect(0, 0, image_gray.cols, image_gray.cols));
	bool ret=false;
	int input_width = 88,input_height=88;
	
	cv::Size baseSize = cv::Size(input_width, input_height);  //Size(656, 368);
	float scale = 0;
	//float th = 1.1;
	//cv::Mat image = cv::Mat::zeros(cv::Size(simage.cols*th, simage.rows*th), simage.type());
	//image(cv::Rect((image.cols - simage.cols) / 2, (image.rows - simage.rows) / 2, simage.cols, simage.rows)) = simage;
	cv::Mat im = getImage(image, baseSize, &scale);
	cv::Mat raw_img = im;

	ncnn::Mat in=ncnn::Mat::from_pixels(raw_img.data,ncnn::Mat::PIXEL_RGB,input_width,input_height);
	ncnn::Mat out;
	const float mean_vals[3] = { 127.5f, 127.5f, 127.5f };
	const float norm_vals[3] = { 1.0 / 255.0,1.0 / 255.0,1.0 / 255.0 };
	in.substract_mean_normalize(mean_vals, norm_vals);//mobile not need mean

	ncnn::Extractor ex = nose_arm_mark.create_extractor();
	ex.set_light_mode(true);
	ex.set_num_threads(4);
	//ex.input("data", in);

	//ex.extract("MConv_Stage7_concat", out);
	ex.input("image", in);
	ex.extract("net_output", out);


	BlobData* nms_out = createBlob_local(1, out.c-1, POSE_MAX_PEOPLE + 1, 3);
	BlobData* input = createBlob_local(1, out.c, baseSize.height, baseSize.width);
	vector<float> keypoints;
	vector<int> shape;
	vector<cv::Mat> input_channels;
	BlobData* net_output = createBlob_local(1, out.c, out.h, out.w);
	for (int i = 0; i < out.c; ++i) {
		cv::Mat um(baseSize.height, baseSize.width, CV_32F, input->list + i*baseSize.height*baseSize.width);
		cv::resize(cv::Mat(net_output->height, net_output->width, CV_32F, out.channel(i).data), um, baseSize, 0, 0, CV_INTER_CUBIC);
		//char name[20];
		//sprintf_s(name, "%02d.jpg", i);
		//imwrite(name, um*500);
	}
	//��ȡÿ��feature map�ľֲ�����ֵ
	nms(input, nms_out, 0.05);
	//with xn_posenms(input, nms_out, 0.15); 
	//connectBodyPartsCpu(keypoints, input->list, nms_out->list, baseSize, POSE_MAX_PEOPLE, 7, 0.2, 7, 0.14, 1, shape);

	connectBodyPartsCpu(keypoints, input->list, nms_out->list, baseSize, POSE_MAX_PEOPLE, 9, 0.05, 6, 0.4, 1, shape);
	//messagefile<<"Has "<<shape[0]<<"peopel"<<std::endl;
	//if(shape.size()>0)
		act_peonum_temp=shape[0];
		// with xn_poseret = renderKeypointsCpu(image, keypoints, shape, 0.053, scale, nose_p_from_keypoint_temp, rear_angle, turn_face, turn_ear);

	int numberKeypoints = shape[1];

	for (auto person = 0; person < act_peonum_temp; person++)
	{
		bool eye_and_shouder = true;
		vector<cv::Point2f> eye_and_shouder_point(EYESHOUDER_RENDER.size());
		for (size_t pn = 0; pn<EYESHOUDER_RENDER.size(); pn += 1)
		{
			const int index1 = (person*numberKeypoints + EYESHOUDER_RENDER[pn])*shape[2];
			if (keypoints[index1 + 2] > 0.2) {
				eye_and_shouder_point[pn]=cv::Point2f(keypoints[index1],keypoints[index1 + 1]);
			}
			else {
				eye_and_shouder = false;
				//return false;
			}
		}
		if (eye_and_shouder) {
			float angleshouder = atan2f((eye_and_shouder_point[3].y - eye_and_shouder_point[2].y), (eye_and_shouder_point[3].x - eye_and_shouder_point[2].x));
			float angleeye = atan2f((eye_and_shouder_point[1].y - eye_and_shouder_point[0].y), (eye_and_shouder_point[1].x - eye_and_shouder_point[0].x));
			cv::Mat rotate = cv::getRotationMatrix2D(eye_and_shouder_point[4], (angleeye - angleshouder) * 180 / M_PI, 1);
			for (size_t pn = 0; pn<FACE_RENDER.size(); pn += 1)
			{
				const int index1 = (person*numberKeypoints + FACE_RENDER[pn])*shape[2];
				cv::Mat p = (cv::Mat_<double>(3, 1) << double(keypoints[index1]), double(keypoints[index1 + 1]), 1.0);
				cv::Mat rp = rotate*p;
				keypoints[index1] = float(rp.at<double>(0));
				keypoints[index1+1] =float( rp.at<double>(1));
			}
		}
		//for (size_t pn = 0; pn<POSE_COCO_PAIRS_RENDER.size(); pn += 2)
		//{
		//	const int index1 = (person*numberKeypoints + POSE_COCO_PAIRS_RENDER[pn])*shape[2];
		//	const int index2 = (person*numberKeypoints + POSE_COCO_PAIRS_RENDER[pn + 1])*shape[2];
		//	if (keypoints[index1+2]>0.05&&keypoints[index2+2]>0.05)
		//	{
		//		const cv::Point kp1(intRound(keypoints[index1] * scale), intRound(keypoints[index1 + 1] * scale));
		//		const cv::Point kp2(intRound(keypoints[index2] * scale), intRound(keypoints[index2 + 1] * scale));
		//		line(image, kp1, kp2, cv::Scalar(0, 128, 255), 2);
		//	}
		//}
	}

	/*
	//renderKeypointsCpu(image, keypoints, shape, 0.05, scale,nose_p_from_keypoint_temp,rear_angle);
	imshow("act", image);
	cv::waitKey(50);*/
	ret = renderKeypointsCpu(image, keypoints, shape, 0.05, scale, nose_p_from_keypoint_temp, rear_angle, turn_face, turn_ear, left_point_detection);

	//log_write("the nose at(%3.2f,%3.2f)\n",nose_p_from_keypoint_temp.x,nose_p_from_keypoint_temp.y);
 	releaseBlob_local(&net_output);
	releaseBlob_local(&input);
	releaseBlob_local(&nms_out);
	return true;
}
bool key_people_num(cv::Mat image_gray,int & act_peonum_temp,const char* deploy,const char* weight)
{
	ncnn::Net peo_num_reg;
	peo_num_reg.load_param(deploy);
	peo_num_reg.load_model(weight);
	cv::Mat image;
	cv::cvtColor(image_gray,image,CV_GRAY2BGR);
	//bool ret=false;
	int input_width = 192,input_height=96;
	cv::Size baseSize = cv::Size(input_width, input_height);  //Size(656, 368);
	float scale = 0;
	cv::Mat im = getImage(image, baseSize, &scale);
	cv::Mat raw_img = im;
	ncnn::Mat in=ncnn::Mat::from_pixels(raw_img.data,ncnn::Mat::PIXEL_RGB,input_width,input_height);
	ncnn::Mat out;
	const float mean_vals[3] = { 127.5f, 127.5f, 127.5f };
	const float norm_vals[3] = { 1.0 / 255.0,1.0 / 255.0,1.0 / 255.0 };
	in.substract_mean_normalize(mean_vals, norm_vals);
	ncnn::Extractor ex = peo_num_reg.create_extractor();
	ex.set_light_mode(true);
	ex.set_num_threads(4);
	ex.input("image", in);
	ex.extract("net_output", out);
	BlobData* nms_out = createBlob_local(1, out.c-1, POSE_MAX_PEOPLE + 1, 3);
	BlobData* input = createBlob_local(1, out.c, baseSize.height, baseSize.width);
	vector<float> keypoints;
	vector<int> shape;
	vector<cv::Mat> input_channels;
	BlobData* net_output = createBlob_local(1, out.c, out.h, out.w);
	for (int i = 0; i < out.c; ++i) {
		cv::Mat um(baseSize.height, baseSize.width, CV_32F, input->list + i*baseSize.height*baseSize.width);
		cv::resize(cv::Mat(net_output->height, net_output->width, CV_32F, out.channel(i).data), um, baseSize, 0, 0, CV_INTER_CUBIC);
	}
	//��ȡÿ��feature map�ľֲ�����ֵ
	nms(input, nms_out, 0.05);
	connectBodyPartsCpu(keypoints, input->list, nms_out->list, baseSize, POSE_MAX_PEOPLE, 9, 0.05, 8, 0.4, 1, shape);
	//messagefile<<"totle Has "<<shape[0]<<"peopel"<<std::endl;
	//if(shape.size()>0)
	act_peonum_temp=shape[0];
	int numberKeypoints= shape[1];
	for (auto person = 0; person < act_peonum_temp; person++)
	{
		for(size_t pn=0;pn<POSE_COCO_PAIRS_RENDER.size();pn+=2)
		{
			const int index1 = (person*numberKeypoints + POSE_COCO_PAIRS_RENDER[pn])*shape[2];
			const int index2 = (person*numberKeypoints + POSE_COCO_PAIRS_RENDER[pn + 1])*shape[2];
			if(keypoints[index1]>0.05&&keypoints[index2]>0.05)
			{
				const cv::Point kp1(intRound(keypoints[index1]*scale),intRound(keypoints[index1+1]*scale));
				const cv::Point kp2(intRound(keypoints[index2]*scale),intRound(keypoints[index2+1]*scale));
				line(image, kp1, kp2, cv::Scalar(0, 128, 255), 2);
			}
		}
	}
	//renderKeypointsCpu(image, keypoints, shape, 0.05, scale,nose_p_from_keypoint_temp,rear_angle);
	imwrite("act.jpg",image);
	//log_write("the nose at(%3.2f,%3.2f)\n",nose_p_from_keypoint_temp.x,nose_p_from_keypoint_temp.y);
	releaseBlob_local(&net_output);
	releaseBlob_local(&input);
	releaseBlob_local(&nms_out);
	return true;
}

//init model
void init_nose_arm_mark(const char* deploy,const char* weight)
{
	nose_arm_mark.load_param(deploy);
	nose_arm_mark.load_model(weight);
}
