﻿
// MarsDemarcateDlg.cpp: 实现文件
//

#include "stdafx.h"
#include "MarsDemarcate.h"
#include "MarsDemarcateDlg.h"
#include "afxdialogex.h"
#include <opencv2/opencv.hpp>
#include "camera_calibration.h"
#include "mars_smokephone.h"
#ifdef _DEBUG
#define new DEBUG_NEW
#endif
#define scal 4
cv::Mat face_click;
extern cv::Mat face_temp;
extern ncnn::Net nose_arm_mark;
CSocket cSocket;
CSocket cSocketServer;
extern adas_camera jp6_camera;
int d_state = -1;
int face_state = 0;
std::vector<float> left_turn_data, right_turn_data;
extern int left_c;//num
extern int right_c;
extern bool left_lock;
extern bool right_lock;//click
//extern float change_angle, left_shouder, right_shouder, left_nose, right_nose;
char * videoadd = "rtsp://admin:@192.168.0.169:554/mode=real&idc=1&ids=1";
// 用于应用程序“关于”菜单项的 CAboutDlg 对话框

class CAboutDlg : public CDialogEx
{
public:
	CAboutDlg();

// 对话框数据
#ifdef AFX_DESIGN_TIME
	enum { IDD = IDD_ABOUTBOX };
#endif

	protected:
	virtual void DoDataExchange(CDataExchange* pDX);    // DDX/DDV 支持

// 实现
protected:
	DECLARE_MESSAGE_MAP()
};

CAboutDlg::CAboutDlg() : CDialogEx(IDD_ABOUTBOX)
{
}

void CAboutDlg::DoDataExchange(CDataExchange* pDX)
{
	CDialogEx::DoDataExchange(pDX);
}

BEGIN_MESSAGE_MAP(CAboutDlg, CDialogEx)
END_MESSAGE_MAP()


// CMarsDemarcateDlg 对话框



CMarsDemarcateDlg::CMarsDemarcateDlg(CWnd* pParent /*=nullptr*/)
	: CDialogEx(IDD_MARSDEMARCATE_DIALOG, pParent)
{
	m_hIcon = AfxGetApp()->LoadIcon(IDR_MAINFRAME);
}

void CMarsDemarcateDlg::DoDataExchange(CDataExchange* pDX)
{
	CDialogEx::DoDataExchange(pDX);
	DDX_Control(pDX, IDC_IPADDRESS_IP, ipToD);
	DDX_Control(pDX, IDC_IPADDRESS_IP2, ipToD2);
	DDX_Control(pDX, IDC_EDIT_INFO, editInfo);
	DDX_Control(pDX, IDC_EDIT_FILE_PATH, editFilePath);
	DDX_Control(pDX, IDC_EDIT_LANE_D_L, editLaneDL);
	DDX_Control(pDX, IDC_EDIT_LANE_D_R, editLaneDR);
	
	DDX_Control(pDX, IDC_CHECK_LANE_D_L, cboxLaneLlock);
	DDX_Control(pDX, IDC_CHECK_LANE_D_R, cboxLaneRlock);
	DDX_Control(pDX, IDC_EDIT_HINT, editHint);
	DDX_Control(pDX, IDC_EDIT_LANE_RANGE, editLaneRANGE);
	//DDX_Control(pDX, IDC_EDIT_LSHOUDER, editSHOUDER_L);
	//DDX_Control(pDX, IDC_EDIT_RSHOUDER, editSHOUDER_R);
	DDX_Control(pDX, IDC_EDIT_LNOSE, editNOSE_L);
	DDX_Control(pDX, IDC_EDIT_RNOSE, editNOSE_R);
}

BEGIN_MESSAGE_MAP(CMarsDemarcateDlg, CDialogEx)
	ON_WM_SYSCOMMAND()
	ON_WM_PAINT()
	ON_WM_QUERYDRAGICON()
	ON_BN_CLICKED(IDC_BUTTON_UP, &CMarsDemarcateDlg::OnBnClickedButtonUp)
	ON_BN_CLICKED(IDC_BUTTON_SURE, &CMarsDemarcateDlg::OnBnClickedButtonSure)
	ON_BN_CLICKED(IDC_BUTTON_LANE_DEMARCATE, &CMarsDemarcateDlg::OnBnClickedButtonLaneDemarcate)
	ON_BN_CLICKED(IDC_BUTTON_CONNECT, &CMarsDemarcateDlg::OnBnClickedButtonConnect)
	ON_BN_CLICKED(ID_BUTTON_DEMARCATE, &CMarsDemarcateDlg::OnBnClickedButtonDemarcate)
	ON_BN_CLICKED(IDC_BUTTON_FACE_DEMARCATE2, &CMarsDemarcateDlg::OnBnClickedButtonFaceDemarcate2)
	ON_BN_CLICKED(IDC_BUTTON_DCAB_DEMARCATE3, &CMarsDemarcateDlg::OnBnClickedButtonDcabDemarcate3)
	ON_BN_CLICKED(IDC_BUTTON_LEFT, &CMarsDemarcateDlg::OnBnClickedButtonLeft)
	ON_BN_CLICKED(IDC_BUTTON_RIGHT, &CMarsDemarcateDlg::OnBnClickedButtonRight)
	ON_BN_CLICKED(IDC_BUTTON_DOWN, &CMarsDemarcateDlg::OnBnClickedButtonDown)
	ON_BN_CLICKED(ID_BUTTON_AMP, &CMarsDemarcateDlg::OnBnClickedButtonAmp)
	ON_BN_CLICKED(IDC_BUTTON_SHR, &CMarsDemarcateDlg::OnBnClickedButtonShr)
	ON_BN_CLICKED(IDC_BUTTON_FILE_SELECT, &CMarsDemarcateDlg::OnBnClickedButtonFileSelect)
	ON_BN_CLICKED(IDC_BUTTON_DEMARCATE_END, &CMarsDemarcateDlg::OnBnClickedButtonDemarcateEnd)
	ON_BN_CLICKED(IDC_BUTTON_FILE_SAVE, &CMarsDemarcateDlg::OnBnClickedButtonFileSave)
	ON_BN_CLICKED(IDC_CHECK_LANE_D_L, &CMarsDemarcateDlg::OnBnClickedCheckLaneDL)
	ON_BN_CLICKED(IDC_CHECK_LANE_D_R, &CMarsDemarcateDlg::OnBnClickedCheckLaneDR)
	ON_BN_CLICKED(IDC_BUTTON_FILE_SEND, &CMarsDemarcateDlg::OnBnClickedButtonFileSend)
	ON_NOTIFY(IPN_FIELDCHANGED, IDC_IPADDRESS_IP, &CMarsDemarcateDlg::OnIpnFieldchangedIpaddressIp)
	ON_NOTIFY(IPN_FIELDCHANGED, IDC_IPADDRESS_IP2, &CMarsDemarcateDlg::OnIpnFieldchangedIpaddressIp2)
	ON_EN_CHANGE(IDC_EDIT_LANE_D_L, &CMarsDemarcateDlg::OnEnChangeEditLaneDL)
	
	ON_EN_CHANGE(IDC_EDIT_LANE_D_R, &CMarsDemarcateDlg::OnEnChangeEditLaneDR)
	ON_EN_CHANGE(IDC_EDIT_LANE_RANGE, &CMarsDemarcateDlg::OnEnChangeEditLaneRange)

	//ON_EN_CHANGE(IDC_EDIT_LSHOUDER, &CMarsDemarcateDlg::OnEnChangeEditLshouder)
	//ON_EN_CHANGE(IDC_EDIT_RSHOUDER, &CMarsDemarcateDlg::OnEnChangeEditRshouder)
	ON_EN_CHANGE(IDC_EDIT_LNOSE, &CMarsDemarcateDlg::OnEnChangeEditLnose)
	ON_EN_CHANGE(IDC_EDIT_RNOSE, &CMarsDemarcateDlg::OnEnChangeEditRnose)
	ON_EN_CHANGE(IDC_EDIT_HINT, &CMarsDemarcateDlg::OnEnChangeEditHint)
	ON_BN_CLICKED(IDC_L_FACE, &CMarsDemarcateDlg::OnBnClickedLFace)
	ON_BN_CLICKED(IDC_R_FACE, &CMarsDemarcateDlg::OnBnClickedRFace)
	ON_BN_CLICKED(IDC_CANNCEL, &CMarsDemarcateDlg::OnBnClickedCanncel)
	ON_BN_CLICKED(IDC_CAL, &CMarsDemarcateDlg::OnBnClickedCal)
END_MESSAGE_MAP()


// CMarsDemarcateDlg 消息处理程序

wchar_t* UTF8ToUnicode(const char* str)
{
	int    textlen;
	wchar_t* result;
	textlen = MultiByteToWideChar(CP_UTF8, 0, str, -1, NULL, 0);
	result = (wchar_t*)malloc((textlen + 1) * sizeof(wchar_t));
	memset(result, 0, (textlen + 1) * sizeof(wchar_t));
	MultiByteToWideChar(CP_UTF8, 0, str, -1, (LPWSTR)result, textlen);
	return    result;
}

BOOL CMarsDemarcateDlg::OnInitDialog()
{
	CDialogEx::OnInitDialog();

	// 将“关于...”菜单项添加到系统菜单中。

	// IDM_ABOUTBOX 必须在系统命令范围内。
	ASSERT((IDM_ABOUTBOX & 0xFFF0) == IDM_ABOUTBOX);
	ASSERT(IDM_ABOUTBOX < 0xF000);

	CMenu* pSysMenu = GetSystemMenu(FALSE);
	if (pSysMenu != nullptr)
	{
		BOOL bNameValid;
		CString strAboutMenu;
		bNameValid = strAboutMenu.LoadString(IDS_ABOUTBOX);
		ASSERT(bNameValid);
		if (!strAboutMenu.IsEmpty())
		{
			pSysMenu->AppendMenu(MF_SEPARATOR);
			pSysMenu->AppendMenu(MF_STRING, IDM_ABOUTBOX, strAboutMenu);
		}
	}

	// 设置此对话框的图标。  当应用程序主窗口不是对话框时，框架将自动
	//  执行此操作
	SetIcon(m_hIcon, TRUE);			// 设置大图标
	SetIcon(m_hIcon, FALSE);		// 设置小图标

	// TODO: 在此添加额外的初始化代码
	AfxSocketInit();

	if (!cSocket.Create())
	{
		editInfo.ReplaceSel(L"创建失败\r\n");
	}


	if (PathFileExists(L".\\readme.txt"))
	{

		CFile ReadF(L".\\readme.txt", CFile::modeRead);
		TCHAR* temp = new TCHAR[ReadF.GetLength() / 2 + 1];
		ReadF.Read(temp, ReadF.GetLength());
		temp[ReadF.GetLength() / 2] = 0;
		GetDlgItem(IDC_EDIT_HINT)->SetWindowTextW(temp);
		
		
		ReadF.Close();//关闭文件
	}
	ipToD2.SetAddress(htonl(inet_addr("192.168.0.169")));
	ipToD.SetAddress(htonl(inet_addr("192.168.0.100")));
	return TRUE;  // 除非将焦点设置到控件，否则返回 TRUE
}

void CMarsDemarcateDlg::OnSysCommand(UINT nID, LPARAM lParam)
{
	if ((nID & 0xFFF0) == IDM_ABOUTBOX)
	{
		CAboutDlg dlgAbout;
		dlgAbout.DoModal();
	}
	else
	{
		CDialogEx::OnSysCommand(nID, lParam);
	}
}

// 如果向对话框添加最小化按钮，则需要下面的代码
//  来绘制该图标。  对于使用文档/视图模型的 MFC 应用程序，
//  这将由框架自动完成。

void CMarsDemarcateDlg::OnPaint()
{
	if (IsIconic())
	{
		CPaintDC dc(this); // 用于绘制的设备上下文

		SendMessage(WM_ICONERASEBKGND, reinterpret_cast<WPARAM>(dc.GetSafeHdc()), 0);

		// 使图标在工作区矩形中居中
		int cxIcon = GetSystemMetrics(SM_CXICON);
		int cyIcon = GetSystemMetrics(SM_CYICON);
		CRect rect;
		GetClientRect(&rect);
		int x = (rect.Width() - cxIcon + 1) / 2;
		int y = (rect.Height() - cyIcon + 1) / 2;

		// 绘制图标
		dc.DrawIcon(x, y, m_hIcon);
	}
	else
	{
		CDialogEx::OnPaint();
	}
}

//当用户拖动最小化窗口时系统调用此函数取得光标
//显示。
HCURSOR CMarsDemarcateDlg::OnQueryDragIcon()
{
	return static_cast<HCURSOR>(m_hIcon);
}









void CMarsDemarcateDlg::SendDemarcateProto(CSocket &socket, BYTE pro)
{
	BYTE check = 0;
	BYTE proto[7] = {'x','n','i','s','d'};
	proto[5] = pro;
	for (int i = 0; i < 6; i++) {
		check += proto[i];
	}
	proto[6] = check;

	socket.Send(proto,7);
}






//标定连接视频流服务器
void CMarsDemarcateDlg::OnBnClickedButtonConnect()
{
	// TODO: 在此添加控件通知处理程序代码
	
	CString strIP;
	
		BYTE nf1, nf2, nf3, nf4;
		int nRcvTimeOut = 10000;
		//cSocket.SetSockOpt(SO_SNDTIMEO, &nRcvTimeOut, sizeof(nRcvTimeOut));
		ipToD.GetAddress(nf1, nf2, nf3, nf4);
		//cSocket.m_nTimeOut = 100;
		strIP.Format(_T("%d.%d.%d.%d"), nf1, nf2, nf3, nf4);//这里的nf得到的值是IP值了
		//转换需要连接的端口内容类型
		int nPort = 8880;

		if (cSocket.Connect(strIP, nPort)){
			editInfo.ReplaceSel(L"连接成功\r\n");
		}
		else {
			editInfo.ReplaceSel(L"连接失败\r\n");
			return;
		}
}
struct d_stream_hdr {
	char hdr[3] = { 'x', 'n', 'd' };
	char lane_num;
	int width;
	int highth;
	int data_length;
};


extern cv::Mat frame_full;
BOOL show_flag = TRUE;
UINT  getStreamFunction(LPVOID  pParam)
{
	CMarsDemarcateDlg* dlg = (CMarsDemarcateDlg*)pParam;
	
	CString strIP2;
	BYTE nnf1, nnf2, nnf3, nnf4;
	dlg->ipToD2.GetAddress(nnf1, nnf2, nnf3, nnf4);
	strIP2.Format(_T("rtsp://admin:@%d.%d.%d.%d:554/mode=real&idc=1&ids=1"), nnf1, nnf2, nnf3, nnf4);//这里的nf得到的值是IP值了
	USES_CONVERSION;
	//函数T2A和W2A均支持ATL和MFC中的字符
	char *videoaddr = T2A(strIP2.GetBuffer(0));
	/*char buff[1024] = { 0 };
	uchar* p_img = NULL;
	CString strIP;
	struct d_stream_hdr d_hdr;
	BYTE nf1, nf2, nf3, nf4;
	dlg->ipToD.GetAddress(nf1, nf2, nf3, nf4);
	strIP.Format(_T("%d.%d.%d.%d"), nf1, nf2, nf3, nf4);//这里的nf得到的值是IP值了
	int nPort = 8888;
	int len = 0;
	int ret;
	AfxSocketInit();*/
	VideoCapture cap;
	cap.open(videoaddr);
	cv::Mat frame_full;
	//cv::Mat show_img;
	d_state = 0;
	show_flag = TRUE;
	
	while(1){
		if (!show_flag)
			break;
		cap.read(frame_full);
		cv::namedWindow("frame_full");
		camera(frame_full);
	}
	cv::destroyAllWindows();
	frame_full.release();
	cap.release();

	return 0;
}
UINT  laneStream(LPVOID  pParam)
{
	CMarsDemarcateDlg* dlg = (CMarsDemarcateDlg*)pParam;
	CString strIP2;
	BYTE nnf1, nnf2, nnf3, nnf4;
	dlg->ipToD2.GetAddress(nnf1, nnf2, nnf3, nnf4);
	strIP2.Format(_T("rtsp://admin:@%d.%d.%d.%d:554/mode=real&idc=1&ids=1"), nnf1, nnf2, nnf3, nnf4);//这里的nf得到的值是IP值了
	USES_CONVERSION;
	//函数T2A和W2A均支持ATL和MFC中的字符
	char *videoaddr = T2A(strIP2.GetBuffer(0));
	VideoCapture cap;
	cap.open(videoaddr);
	d_state = 1;
	show_flag = TRUE;
	cv::Mat frame_full;
	
	while (1) {
		if (!show_flag)
			break;
		cap.read(frame_full);
		cv::namedWindow("lane");
		lane_camera_box(frame_full);


	}
	frame_full.release();
	cap.release();
	cv::destroyAllWindows();
	return 0;
}

UINT  faceStream(LPVOID  pParam)
{
	CMarsDemarcateDlg* dlg = (CMarsDemarcateDlg*)pParam;

	CString strIP2;
	BYTE nnf1, nnf2, nnf3, nnf4;
	dlg->ipToD2.GetAddress(nnf1, nnf2, nnf3, nnf4);
	strIP2.Format(_T("rtsp://admin:@%d.%d.%d.%d:554/mode=real&idc=1&ids=1"), nnf1, nnf2, nnf3, nnf4);//这里的nf得到的值是IP值了
	USES_CONVERSION;
	//函数T2A和W2A均支持ATL和MFC中的字符
	char *videoaddr = T2A(strIP2.GetBuffer(0));
	VideoCapture cap;
	cap.open(videoaddr);
	d_state = 2;
	show_flag = TRUE;
	cv::Mat frame_full;
	
	while (1) {
		if (!show_flag)
			break;
		cap.read(frame_full);
		cv::namedWindow("face_show");

		face_camera(frame_full);


	}
	frame_full.release();
	cap.release();
	cv::destroyAllWindows();
	return 0;
}

UINT  peonumStream(LPVOID  pParam)
{
	CMarsDemarcateDlg* dlg = (CMarsDemarcateDlg*)pParam;

	CString strIP2;
	BYTE nnf1, nnf2, nnf3, nnf4;
	dlg->ipToD2.GetAddress(nnf1, nnf2, nnf3, nnf4);
	strIP2.Format(_T("rtsp://admin:@%d.%d.%d.%d:554/mode=real&idc=1&ids=1"), nnf1, nnf2, nnf3, nnf4);//这里的nf得到的值是IP值了
	USES_CONVERSION;
	//函数T2A和W2A均支持ATL和MFC中的字符
	char *videoaddr = T2A(strIP2.GetBuffer(0));
	VideoCapture cap;
	cap.open(T2A(strIP2.GetBuffer(0)));
	d_state = 3;
	show_flag = TRUE;
	cv::Mat frame_full, frame_full_org;
	
	while (1) {
		if (!show_flag)
			break;
		cap.read(frame_full_org);
		cv::namedWindow("peo_show");
		Mat rotate = getRotationMatrix2D(Point(jp6_camera._center_x, jp6_camera._center_y), -16, 1);
		warpAffine(frame_full_org, frame_full, rotate, Size(1632, 1632));
		peonum_camera(frame_full);
		

	}
	frame_full.release();

	cap.release();
	cv::destroyAllWindows();
	return 0;
}

void CMarsDemarcateDlg::OnBnClickedButtonDemarcate()
{
	// TODO: 在此添加控件通知处理程序代码
	//SendDemarcateProto(cSocket, D_START);
	
	


	if (PathFileExists(L".\\demarcate.cfg"))
	{
		DeleteFile(L".\\demarcate.cfg");
	}
	hThread = CreateThread(NULL,0,(LPTHREAD_START_ROUTINE)getStreamFunction,this,0,&ThreadID);

}

//车道标定
void CMarsDemarcateDlg::OnBnClickedButtonLaneDemarcate()
{
	// TODO: 在此添加控件通知处理程序代码
	hThread = CreateThread(NULL, 0, (LPTHREAD_START_ROUTINE)laneStream, this, 0, &ThreadID);
}
//人脸标定	
void CMarsDemarcateDlg::OnBnClickedButtonFaceDemarcate2()
{
	// TODO: 在此添加控件通知处理程序代码
	init_nose_arm_mark("arm25.param","arm25.bin");
	left_turn_data.clear();
	right_turn_data.clear();
	CString str;
	editLaneRANGE.GetWindowTextW(str);
	this->an = _ttof(str);
	hThread = CreateThread(NULL, 0, (LPTHREAD_START_ROUTINE)faceStream, this, 0, &ThreadID);

}

//驾驶室标定
void CMarsDemarcateDlg::OnBnClickedButtonDcabDemarcate3()
{
	// TODO: 在此添加控件通知处理程序代码
	hThread = CreateThread(NULL, 0, (LPTHREAD_START_ROUTINE)peonumStream, this, 0, &ThreadID);
	
}
#define __UP 0
#define __DOWN 1
#define __LEFT 2
#define __RIGHT 3
#define __AMP 4
#define __SHR 5
char check_camera(int mode)
{
	char res = -1;
	switch (mode) {
	case 0:
		if (jp6_camera._center_x + jp6_camera._radius + scal >= videoWidth)
			return __RIGHT;
		if(jp6_camera._center_x - jp6_camera._radius - scal < 0)
			return __LEFT;
		if (jp6_camera._center_y + jp6_camera._radius + scal >= videoHeight)
			return __DOWN;
		if(jp6_camera._center_y - jp6_camera._radius - scal < 0)
			return __UP;
		if (jp6_camera._radius < 100)
			return __SHR;
		break;
	case 2:
		if (jp6_camera._face_x + jp6_camera._face_width + scal >= videoWidth)
			return __RIGHT;
		if( jp6_camera._face_x - scal < 0)
			return __LEFT;
		if (jp6_camera._face_y + jp6_camera._face_height + scal >= videoHeight)
			return __DOWN; 
		if( jp6_camera._face_y - scal < 0)
			return __UP;
		if (min(jp6_camera._face_height, jp6_camera._face_width) < 200)
			return __SHR;
		if (jp6_camera._face_y + jp6_camera._face_height*1.1 + scal >= videoHeight || jp6_camera._face_x + jp6_camera._face_width*1.1 + scal >= videoWidth)
			return __AMP;	
		break;
	case 3:
		if (jp6_camera._peo_num_x + jp6_camera._peo_num_width + scal >= videoWidth)
			return __RIGHT; 
		if( jp6_camera._peo_num_x - scal < 0)
			return __LEFT;
		if (jp6_camera._peo_num_y + jp6_camera._peo_num_height + scal >= videoHeight)
			return __DOWN; 
		if(jp6_camera._peo_num_y - scal < 0)
			return __UP;
		if (min(jp6_camera._peo_num_width, jp6_camera._peo_num_height) < 200)
			return __SHR;
		if (jp6_camera._peo_num_y + jp6_camera._peo_num_height*1.1 + scal >= videoHeight || jp6_camera._peo_num_x + jp6_camera._peo_num_width*1.1 + scal >= videoWidth)
			return __AMP;
		break;
	case 1:
		if (jp6_camera._lane_x + jp6_camera._lane_width + scal >= videoWidth)
			return __RIGHT; 
		if( jp6_camera._lane_x - scal < 0)
			return __LEFT;
		if (min(jp6_camera._lane_height, jp6_camera._lane_width) < 50)
			return __SHR;
		if(max(jp6_camera._lane_height, jp6_camera._lane_width) > 450)
			return __AMP;
		if (jp6_camera._lane_y + jp6_camera._lane_height + scal >= videoHeight / 4)
			return __DOWN;
		if( jp6_camera._lane_y - scal < 0)
			return __UP;
		break;
	default:break;
	}
	return res;
}

void CMarsDemarcateDlg::OnBnClickedButtonUp()
{
	// TODO: 在此添加控件通知处理程序代码
	int t;
	int res = check_camera(d_state);
	if (res == __UP)
		return;
	// TODO: 在此添加控件通知处理程序代码
	switch (d_state) {
	case 0:
		jp6_camera._center_y -= scal;
		break;
	case 1:
		t = jp6_camera._lane_y;
		t = t - scal;
		t = t / 4 * 4;
		jp6_camera._lane_y = t;
		break;
	case 2:
		jp6_camera._face_y -= scal;
		break;
	case 3:
		jp6_camera._peo_num_y -= scal;
		break;
	default:break;
	}
}

//左移
void CMarsDemarcateDlg::OnBnClickedButtonLeft()
{
	int res = check_camera(d_state);
	if (res == __LEFT)
		return;
	int t;
	// TODO: 在此添加控件通知处理程序代码
	switch (d_state) {
	case 0:
		jp6_camera._center_x -= scal;
		break;
	case 1:
		t = jp6_camera._lane_x;
		t = t - scal;
		t = t / 4 * 4;
		jp6_camera._lane_x = t;
		break;
	case 2:
		jp6_camera._face_x -= scal;
		break;
	case 3:
		jp6_camera._peo_num_x -= scal;
		break;
	default:break;
	}
}

//
void CMarsDemarcateDlg::OnBnClickedButtonRight()
{
	int t;
	int res = check_camera(d_state);
	if (res == __RIGHT)
		return;
	// TODO: 在此添加控件通知处理程序代码
	switch (d_state) {
	case 0:
		jp6_camera._center_x += scal;
		break;
	case 1:
		t = jp6_camera._lane_x;
		t = t + scal;
		t = t / 4 * 4;
		jp6_camera._lane_x = t;
		break;
	case 2:
		jp6_camera._face_x += scal;
		break;
	case 3:
		jp6_camera._peo_num_x += scal;
		break;
	default:break;
	}
}


void CMarsDemarcateDlg::OnBnClickedButtonDown()
{
	int res = check_camera(d_state);
	if (res == __DOWN)
		return;
	// TODO: 在此添加控件通知处理程序代码
	int t;
	switch (d_state) {
	case 0:
		jp6_camera._center_y += scal;
		break;
	case 1:
		t = jp6_camera._lane_y;
		t = t + scal;
		t = t / 4 * 4;
		jp6_camera._lane_y = t;
		break;
	case 2:
		jp6_camera._face_y += scal;
		break;
	case 3:
		jp6_camera._peo_num_y += scal;
		break;
	default:break;
	}
}


void CMarsDemarcateDlg::OnBnClickedButtonAmp()
{
	// TODO: 在此添加控件通知处理程序代码
	int res = check_camera(d_state);
	if (res == __AMP)
		return;
	int t, t2;
	switch (d_state) {
	case 0:
		jp6_camera._radius *= 1.1;
		break;
	case 1:
		t = jp6_camera._lane_width;
		t = t*1.1;
		t = t / 4 * 4;
		jp6_camera._lane_width = t;
		
		jp6_camera._lane_height *= 1.1;

		break;
	case 2:
		if (res == __DOWN || res == __RIGHT)
			break;
		jp6_camera._face_width *= 1.1;
		jp6_camera._face_height *= 1.1;
		break;
	case 3:
		if (res == __DOWN || res == __RIGHT)
			break;
		jp6_camera._peo_num_width *= 1.1;
		jp6_camera._peo_num_height *= 1.1;
		break;
	default:break;
	}
}


void CMarsDemarcateDlg::OnBnClickedButtonShr()
{
	int t;
	int res = check_camera(d_state);
	if (res == __SHR)
		return;
	switch (d_state) {
	case 0:
		jp6_camera._radius /= 1.1;
		break;
	case 1:
		t = jp6_camera._lane_width;
		t = t / 1.1;
		t = t / 4 * 4;
		jp6_camera._lane_width = t;
		jp6_camera._lane_height /= 1.1;
		break;
	case 2:
		jp6_camera._face_width /= 1.1;
		jp6_camera._face_height /= 1.1;
		break;
	case 3:
		jp6_camera._peo_num_width /= 1.1;
		jp6_camera._peo_num_height /= 1.1;
		break;
	default:break;
	}
}


void CMarsDemarcateDlg::OnBnClickedButtonSure()
{
	// TODO: 在此添加控件通知处理程序代码
	if(d_state == 0)
		CreateDstmap(jp6_camera._center_x, jp6_camera._center_y, jp6_camera._radius, videoHeight, videoWidth, videoHeight, videoWidth);
	if (d_state == 1)
	{
		CString str;
		editLaneRANGE.GetWindowTextW(str);
		jp6_camera._change_angle= _ttof(str);
		//change_angle = _ttof(str);
	}
	if (d_state == 2)
	{
		/*CString strsl, strsr, strnl, strnr;
		editSHOUDER_L.GetWindowTextW(strsl);
		float shouderl= _ttof(strsl);
		editSHOUDER_R.GetWindowTextW(strsr);
		float shouderr = _ttof(strsr);
		editNOSE_L.GetWindowTextW(strnl);
		float nosel = _ttof(strnl);
		editNOSE_R.GetWindowTextW(strnr);
		float noser = _ttof(strnr);

		

		if (shouderr > shouderl&&noser> shouderl) {
			jp6_camera._left_turn = ( nosel- shouderl ) / (shouderr - shouderl)-0.15;
			jp6_camera._right_turn = ( noser- shouderl ) / (shouderr - shouderl);
		}
		*/
		char buf[20];
		float mean_left = 0, std_left = 0, mean_right = 0, std_right = 0;
		editNOSE_L.SetSel(0, -1);//清空显示
		editNOSE_R.SetSel(0, -1);
		if (left_turn_data.size() > 0)
		{
			for (size_t i = 0; i < left_turn_data.size(); i++)//均值
				mean_left += left_turn_data[i];
			jp6_camera._left_turn = mean_left / left_turn_data.size();

			for (size_t i = 0; i < left_turn_data.size(); i++)//方差
			{
				std_left += (jp6_camera._left_turn - left_turn_data[i])*(jp6_camera._left_turn - left_turn_data[i]);
			}
			std_left /= left_turn_data.size();

			sprintf_s(buf, " %0.4f ", std_left);
			int num = MultiByteToWideChar(0, 0, buf, -1, NULL, 0);
			wchar_t *wide = new wchar_t[num];

			MultiByteToWideChar(0, 0, buf, -1, wide, num);
			
			editNOSE_L.ReplaceSel(wide);

			delete[] wide;
		}
		else
			editNOSE_L.ReplaceSel(L"no left face");
		if (right_turn_data.size() > 0)
		{
			for (size_t i = 0; i < right_turn_data.size(); i++)//均值
				mean_right += right_turn_data[i];
			jp6_camera._right_turn = mean_right / right_turn_data.size();

			for (size_t i = 0; i < right_turn_data.size(); i++)//方差
			{
				std_right += (jp6_camera._right_turn - right_turn_data[i])*(jp6_camera._right_turn - right_turn_data[i]);
			}
			std_right /= right_turn_data.size();

			sprintf_s(buf, " %0.4f ", std_right);
			int num = MultiByteToWideChar(0, 0, buf, -1, NULL, 0);
			wchar_t *wider = new wchar_t[num];

			MultiByteToWideChar(0, 0, buf, -1, wider, num);

			editNOSE_R.ReplaceSel(wider);

			delete[] wider;
		}
		else
			editNOSE_R.ReplaceSel(L"no right face");



		nose_arm_mark.clear();
	}

	show_flag = FALSE;
	CloseHandle(hThread);
}


void CMarsDemarcateDlg::OnBnClickedButtonFileSelect()
{
	// TODO: 在此添加控件通知处理程序代码
	CFileDialog dlg(TRUE, NULL, NULL, OFN_ALLOWMULTISELECT | OFN_HIDEREADONLY | OFN_FILEMUSTEXIST, NULL, NULL);
	dlg.m_ofn.lpstrTitle = _T("选择文件");
	CString filename;

	if (dlg.DoModal() == IDOK)
	{
		POSITION fileNamesPosition = dlg.GetStartPosition();
		while (fileNamesPosition != NULL)
		{
			filename = dlg.GetNextPathName(fileNamesPosition);
			editFilePath.SetWindowTextW(filename);
		}

	}

}


void CMarsDemarcateDlg::OnBnClickedButtonDemarcateEnd()
{
	// TODO: 在此添加控件通知8处理程序代码
	cSocket.Send("xnisdend",8 );
	editInfo.ReplaceSel(L"标定结束\r\n");
	cSocket.Close();
}


void CMarsDemarcateDlg::OnBnClickedButtonFileSave()
{
	// TODO: 在此添加控件通知处理程序代码
	save_config(&jp6_camera);
}


void CMarsDemarcateDlg::OnBnClickedCheckLaneDL()
{
	// TODO: 在此添加控件通知处理程序代码
	CString str;
	editLaneDL.GetWindowTextW(str);

	left_c = _ttoi(str);
	//str.Format(_T("%d"), left_c);
	if (cboxLaneLlock.GetCheck()) {
		left_lock = true;
		editLaneDL.EnableWindow(FALSE);
	}
	else {
		left_lock = false;
		editLaneDL.EnableWindow(TRUE);
	}

}


void CMarsDemarcateDlg::OnBnClickedCheckLaneDR()
{
	// TODO: 在此添加控件通知处理程序代码
	CString str;
	editLaneDR.GetWindowTextW(str);
	//str.Format(_T("%d"), right_c);
	right_c = _ttoi(str);
	if (cboxLaneRlock.GetCheck()) {
		right_lock = true;
		editLaneDR.EnableWindow(FALSE);
	}else {
		right_lock = false;
		editLaneDR.EnableWindow(TRUE);
	}
}


int CMarsDemarcateDlg:: save_config(struct adas_camera * p)
{
	CString strIP2;
	BYTE nnf1, nnf2, nnf3, nnf4;
	ipToD2.GetAddress(nnf1, nnf2, nnf3, nnf4);
	strIP2.Format(_T("%d.%d.%d.%d"), nnf1, nnf2, nnf3, nnf4);//这里的nf得到的值是IP值了
	USES_CONVERSION;
	//函数T2A和W2A均支持ATL和MFC中的字符
	char *videoaddr = T2A(strIP2.GetBuffer(0));

	FILE* fp = NULL;
	fopen_s(&fp,".\\demarcate.cfg","w");
	if (fp == NULL)
		return -1;
	fprintf(fp, "video_addr=%s\n", videoaddr);
	fprintf(fp, "_center_x=%d\n", p->_center_x);
	fprintf(fp, "_cneter_y=%d\n", p->_center_y);
	fprintf(fp, "_radius=%d\n", p->_radius);
	fprintf(fp, "_lane_x=%d\n", p->_lane_x);
	fprintf(fp, "_lane_y=%d\n", p->_lane_y);
	fprintf(fp, "_lane_width=%d\n", p->_lane_width);
	fprintf(fp, "_lane_height=%d\n", p->_lane_height);
	fprintf(fp, "_face_x=%d\n", p->_face_x);
	fprintf(fp, "_face_y=%d\n", p->_face_y);
	fprintf(fp, "_face_width=%d\n", p->_face_width);
	fprintf(fp, "_face_height=%d\n", p->_face_height);
	fprintf(fp, "_peo_num_x=%d\n", p->_peo_num_x);
	fprintf(fp, "_peo_num_y=%d\n", p->_peo_num_y);
	fprintf(fp, "_peo_num_width=%d\n", p->_peo_num_width);
	fprintf(fp, "_peo_num_height=%d\n", p->_peo_num_height);
	fprintf(fp, "_left_point_x=%f\n", p->_left_point_x);
	fprintf(fp, "_left_point_y=%f\n", p->_left_point_y);
	fprintf(fp, "_left_angle=%f\n", p->_left_angle);
	fprintf(fp, "_right_point_x=%f\n", p->_right_point_x);
	fprintf(fp, "_right_point_y=%f\n", p->_right_point_y);
	fprintf(fp, "_right_angle=%f\n", p->_right_angle);
	fprintf(fp, "_double_lane_dis=%f\n", p->_double_lane_dis);
	fprintf(fp, "_change_angle=%f\n", p->_change_angle);
	fprintf(fp, "_left_turn=%f\n", p->_left_turn);
	fprintf(fp, "_right_turn=%f\n", p->_right_turn);
	fprintf(fp, "_left_weight=%f\n", p->_left_weight);
	fprintf(fp, "_left_bias=%f\n", p->_left_bias);
	fprintf(fp, "_right_weight=%f\n", p->_right_weight);
	fprintf(fp, "_right_bias=%f\n", p->_right_bias);
	fclose(fp);
	return 0;
}


void threadTransferFile(CMarsDemarcateDlg &dlg)
{
	AfxSocketInit();
	char hdr[72] = { "xnis" };
	int n = dlg.strDemarcatefile.ReverseFind('\\') + 1;
	CString strPath = dlg.strDemarcatefile.Right(dlg.strDemarcatefile.GetLength() - n);
	WideCharToMultiByte(CP_ACP, 0, strPath, strPath.GetLength(), &hdr[4], 64, NULL, NULL);

	UINT check = 0, size = 0;
	CFileStatus fileStatus;
	CFile::GetStatus(dlg.strDemarcatefile, fileStatus);
	size = fileStatus.m_size;

	memcpy(&hdr[68], &size, 4);

	cSocket.Send(hdr, 72);

	char revData[1024] = { 0 };
	//接收服务器发送回来的内容(该方法会阻塞, 在此等待有内容接收到才继续向下执行)
	cSocket.Receive(revData, 1024);

	if (strcmp(revData, "ready go")) {
		dlg.editInfo.ReplaceSel(L"文件错误\r\n");
		cSocket.Close();
		return;
	}
	dlg.editInfo.ReplaceSel(L"开始文件传输...\r\n");
	HANDLE hFile = CreateFile(dlg.strDemarcatefile, GENERIC_READ, FILE_SHARE_READ, nullptr, OPEN_EXISTING, FILE_FLAG_SEQUENTIAL_SCAN, nullptr);
	if (hFile == INVALID_HANDLE_VALUE)
	{
		cSocket.Close();
		return;
	}
	HANDLE hFileMapping = CreateFileMapping(hFile, nullptr, PAGE_READONLY, 0, 0, nullptr);
	if (!hFileMapping)
	{
		cSocket.Close();
		CloseHandle(hFile);
		return;
	}
	char* pBuffer = (char*)MapViewOfFile(hFileMapping, FILE_MAP_READ, 0, 0, 0);
	if (!pBuffer)
	{
		cSocket.Close();
		CloseHandle(hFileMapping);
		CloseHandle(hFile);
		return;
	}
	char* end = pBuffer + size;
	for (;;) {
		if (pBuffer + 1024 < end) {
			cSocket.Send(pBuffer, 1024);
			pBuffer += 1024;
		}
		else {
			cSocket.Send(pBuffer, end - pBuffer);
			break;
		}
		//memset(revData, 0, 1024);
		//cSocket.Receive(revData, 1024);
	}
	memset(revData, 0, 1024);
	cSocket.Receive(revData, 1024);
	if (strcmp(revData, "ok")) {
		dlg.editInfo.ReplaceSel(L"文件校验失败，请重新升级\r\n");
	}
	else {
		dlg.editInfo.ReplaceSel(L"文件传输完成，请耐心等待升级完成...\r\n");
	}

	cSocket.Close();
	UnmapViewOfFile(pBuffer);
	//关闭文件映射对象
	CloseHandle(hFileMapping);
	//关闭文件对象
	CloseHandle(hFile);
}




void CMarsDemarcateDlg::OnBnClickedButtonFileSend()
{
	// TODO: 在此添加控件通知处理程序代码
	editFilePath.GetWindowTextW(strDemarcatefile);
	if (strDemarcatefile.IsEmpty()) {
		AfxMessageBox(_T("未选择传输文件！"));
		return;
	}

	char hdr[72] = { "xnis" };
	int n = strDemarcatefile.ReverseFind('\\') + 1;
	CString strPath = strDemarcatefile.Right(strDemarcatefile.GetLength() - n);
	WideCharToMultiByte(CP_ACP, 0, strPath, strPath.GetLength(), &hdr[4], 64, NULL, NULL);

	UINT check = 0, size = 0;
	CFileStatus fileStatus;
	CFile::GetStatus(strDemarcatefile, fileStatus);
	size = fileStatus.m_size;

	memcpy(&hdr[68], &size, 4);

	cSocket.Send(hdr, 72);

	char revData[1024] = { 0 };
	//接收服务器发送回来的内容(该方法会阻塞, 在此等待有内容接收到才继续向下执行)
	cSocket.Receive(revData, 1024);

	if (strcmp(revData, "ready go")) {
		editInfo.ReplaceSel(L"文件错误\r\n");
		cSocket.Close();
		return;
	}
	editInfo.ReplaceSel(L"开始文件传输...\r\n");
	HANDLE hFile = CreateFile(strDemarcatefile, GENERIC_READ, FILE_SHARE_READ, nullptr, OPEN_EXISTING, FILE_FLAG_SEQUENTIAL_SCAN, nullptr);
	if (hFile == INVALID_HANDLE_VALUE)
	{
		cSocket.Close();
		return;
	}
	HANDLE hFileMapping = CreateFileMapping(hFile, nullptr, PAGE_READONLY, 0, 0, nullptr);
	if (!hFileMapping)
	{
		cSocket.Close();
		CloseHandle(hFile);
		return;
	}
	char* pBuffer = (char*)MapViewOfFile(hFileMapping, FILE_MAP_READ, 0, 0, 0);
	if (!pBuffer)
	{
		cSocket.Close();
		CloseHandle(hFileMapping);
		CloseHandle(hFile);
		return;
	}
	char* end = pBuffer + size;
	for (;;) {
		if (pBuffer + 1024 < end) {
			cSocket.Send(pBuffer, 1024);
			pBuffer += 1024;
		}
		else {
			cSocket.Send(pBuffer, end - pBuffer);
			break;
		}
		//memset(revData, 0, 1024);
		//cSocket.Receive(revData, 1024);
	}
	memset(revData, 0, 1024);
	cSocket.Receive(revData, 1024);
	if (strcmp(revData, "ok")) {
		editInfo.ReplaceSel(L"文件校验失败，请重新升级\r\n");
	}
	else {
		editInfo.ReplaceSel(L"文件传输完成\r\n");
	}

	//cSocket.Close();
	UnmapViewOfFile(pBuffer);
	//关闭文件映射对象
	CloseHandle(hFileMapping);
	//关闭文件对象
	CloseHandle(hFile);
	//threadTransferFile(this);
	//HANDLE hThread;
	//DWORD ThreadID;
	//hThread = CreateThread(NULL, 0, (LPTHREAD_START_ROUTINE)threadTransferFile, this, 0, &ThreadID);

	//起线程 禁止操作
}


void CMarsDemarcateDlg::OnIpnFieldchangedIpaddressIp(NMHDR *pNMHDR, LRESULT *pResult)
{
	LPNMIPADDRESS pIPAddr = reinterpret_cast<LPNMIPADDRESS>(pNMHDR);
	// TODO: 在此添加控件通知处理程序代码
	*pResult = 0;
	ipTOD_in = true;
	
}


void CMarsDemarcateDlg::OnIpnFieldchangedIpaddressIp2(NMHDR *pNMHDR, LRESULT *pResult)
{
	LPNMIPADDRESS pIPAddr = reinterpret_cast<LPNMIPADDRESS>(pNMHDR);
	// TODO: 在此添加控件通知处理程序代码
	*pResult = 0;
}


void CMarsDemarcateDlg::OnEnChangeEditLaneDL()
{
	// TODO:  如果该控件是 RICHEDIT 控件，它将不
	// 发送此通知，除非重写 CDialogEx::OnInitDialog()
	// 函数并调用 CRichEditCtrl().SetEventMask()，
	// 同时将 ENM_CHANGE 标志“或”运算到掩码中。

	// TODO:  在此添加控件通知处理程序代码
}


void CMarsDemarcateDlg::OnEnChangeEditLaneRange()
{
	// TODO:  如果该控件是 RICHEDIT 控件，它将不
	// 发送此通知，除非重写 CDialogEx::OnInitDialog()
	// 函数并调用 CRichEditCtrl().SetEventMask()，
	// 同时将 ENM_CHANGE 标志“或”运算到掩码中。

	// TODO:  在此添加控件通知处理程序代码
}


void CMarsDemarcateDlg::OnEnChangeEditLaneDR()
{
	// TODO:  如果该控件是 RICHEDIT 控件，它将不
	// 发送此通知，除非重写 CDialogEx::OnInitDialog()
	// 函数并调用 CRichEditCtrl().SetEventMask()，
	// 同时将 ENM_CHANGE 标志“或”运算到掩码中。

	// TODO:  在此添加控件通知处理程序代码
}


void CMarsDemarcateDlg::OnEnChangeEditShouderL()
{
	// TODO:  如果该控件是 RICHEDIT 控件，它将不
	// 发送此通知，除非重写 CDialogEx::OnInitDialog()
	// 函数并调用 CRichEditCtrl().SetEventMask()，
	// 同时将 ENM_CHANGE 标志“或”运算到掩码中。

	// TODO:  在此添加控件通知处理程序代码
}


void CMarsDemarcateDlg::OnEnChangeShouderR()
{
	// TODO:  如果该控件是 RICHEDIT 控件，它将不
	// 发送此通知，除非重写 CDialogEx::OnInitDialog()
	// 函数并调用 CRichEditCtrl().SetEventMask()，
	// 同时将 ENM_CHANGE 标志“或”运算到掩码中。

	// TODO:  在此添加控件通知处理程序代码
}


void CMarsDemarcateDlg::OnEnChangeEditLshouder()
{
	// TODO:  如果该控件是 RICHEDIT 控件，它将不
	// 发送此通知，除非重写 CDialogEx::OnInitDialog()
	// 函数并调用 CRichEditCtrl().SetEventMask()，
	// 同时将 ENM_CHANGE 标志“或”运算到掩码中。

	// TODO:  在此添加控件通知处理程序代码
}


void CMarsDemarcateDlg::OnEnChangeEditRshouder()
{
	// TODO:  如果该控件是 RICHEDIT 控件，它将不
	// 发送此通知，除非重写 CDialogEx::OnInitDialog()
	// 函数并调用 CRichEditCtrl().SetEventMask()，
	// 同时将 ENM_CHANGE 标志“或”运算到掩码中。

	// TODO:  在此添加控件通知处理程序代码
}


void CMarsDemarcateDlg::OnEnChangeEditLnose()
{
	// TODO:  如果该控件是 RICHEDIT 控件，它将不
	// 发送此通知，除非重写 CDialogEx::OnInitDialog()
	// 函数并调用 CRichEditCtrl().SetEventMask()，
	// 同时将 ENM_CHANGE 标志“或”运算到掩码中。

	// TODO:  在此添加控件通知处理程序代码
}


void CMarsDemarcateDlg::OnEnChangeEditRnose()
{
	// TODO:  如果该控件是 RICHEDIT 控件，它将不
	// 发送此通知，除非重写 CDialogEx::OnInitDialog()
	// 函数并调用 CRichEditCtrl().SetEventMask()，
	// 同时将 ENM_CHANGE 标志“或”运算到掩码中。

	// TODO:  在此添加控件通知处理程序代码
}


void CMarsDemarcateDlg::OnEnChangeEditHint()
{
	// TODO:  如果该控件是 RICHEDIT 控件，它将不
	// 发送此通知，除非重写 CDialogEx::OnInitDialog()
	// 函数并调用 CRichEditCtrl().SetEventMask()，
	// 同时将 ENM_CHANGE 标志“或”运算到掩码中。

	// TODO:  在此添加控件通知处理程序代码
}


void CMarsDemarcateDlg::OnBnClickedLFace()
{
	// TODO: 在此添加控件通知处理程序代码
	face_state = 1;//left
	face_click = face_temp.clone();
	imshow("left_face", face_click);
}


void CMarsDemarcateDlg::OnBnClickedRFace()
{
	// TODO: 在此添加控件通知处理程序代码
	face_state = 2;//right
	face_click = face_temp.clone();
	imshow("right_face", face_click);
}


void CMarsDemarcateDlg::OnBnClickedCanncel()
{
	// TODO: 在此添加控件通知处理程序代码
	if (face_state == 1)
		destroyWindow("left_face");
	if (face_state == 2)
		destroyWindow("right_face");
}


void CMarsDemarcateDlg::OnBnClickedCal()
{
	if (face_state == 1) {
		char buf[20];
		float rear_angle,turnface;
		int peo_num;
		Mat gray;
		cvtColor(face_click, gray, CV_BGR2GRAY);
		//cvtColor(gray, gray, CV_GRAY2BGR);
		cv::Point2f p;
		editNOSE_L.SetSel(0, -1);
		
		key_point(gray,p, rear_angle, turnface,peo_num);
		if (peo_num == 1) {
			left_turn_data.push_back(turnface);
			sprintf_s(buf, "%0.4f %d", turnface, left_turn_data.size());
			int num = MultiByteToWideChar(0, 0, buf, -1, NULL, 0);
			wchar_t *wide = new wchar_t[num];

			MultiByteToWideChar(0, 0, buf, -1, wide, num);
			
			editNOSE_L.ReplaceSel(wide);

			//editNOSE_L.ReplaceSel(L"123");

			delete[] wide;
		}
		else
			editNOSE_L.ReplaceSel(L"not one face");

	}
	if (face_state == 2) {
		char buf[20];
		float rear_angle, turnface;
		int peo_num;
		Mat gray;
		cvtColor(face_click, gray, CV_BGR2GRAY);
		//cvtColor(gray, gray, CV_GRAY2BGR);
		cv::Point2f p;
		editNOSE_R.SetSel(0, -1);
		key_point(gray, p, rear_angle, turnface, peo_num);
		if (peo_num == 1) {
			right_turn_data.push_back(turnface);
			sprintf_s(buf, "%0.4f %d", turnface, right_turn_data.size());
			int num = MultiByteToWideChar(0, 0, buf, -1, NULL, 0);
			wchar_t *wide = new wchar_t[num];

			MultiByteToWideChar(0, 0, buf, -1, wide, num);

			editNOSE_R.ReplaceSel(wide);



			delete[] wide;
		}
		else
			editNOSE_R.ReplaceSel(L"not one face");
	}
	if (face_state == 1)
		destroyWindow("left_face");
	if (face_state == 2)
		destroyWindow("right_face");
	// TODO: 在此添加控件通知处理程序代码
}
