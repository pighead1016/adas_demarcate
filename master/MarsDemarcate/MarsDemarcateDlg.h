
// MarsDemarcateDlg.h: 头文件
//

#pragma once
#include <afxsock.h>

// CMarsDemarcateDlg 对话框
class CMarsDemarcateDlg : public CDialogEx
{
// 构造
public:
	CMarsDemarcateDlg(CWnd* pParent = nullptr);	// 标准构造函数

// 对话框数据
#ifdef AFX_DESIGN_TIME
	enum { IDD = IDD_MARSDEMARCATE_DIALOG };
#endif

	protected:
	virtual void DoDataExchange(CDataExchange* pDX);	// DDX/DDV 支持


// 实现
protected:
	HICON m_hIcon;

	// 生成的消息映射函数
	virtual BOOL OnInitDialog();
	afx_msg void OnSysCommand(UINT nID, LPARAM lParam);
	afx_msg void OnPaint();
	afx_msg HCURSOR OnQueryDragIcon();
	DECLARE_MESSAGE_MAP()
public:
	char videoaddr[100];
	HANDLE hThread;
	DWORD ThreadID;
	float an;
	int left_point;
	CString strDemarcatefile;
	enum demarcate_proto { D_START, LANE_D, FACE_D, CAB_D ,UP,DOWN,LEFT,RIGHT,AMP,SHR,SAVE};
	afx_msg void OnBnClickedButtonUp();
	afx_msg void OnBnClickedButtonSure();
	afx_msg void OnBnClickedButtonLaneDemarcate();
	afx_msg void OnBnClickedButtonConnect();
	void CMarsDemarcateDlg::SendDemarcateProto(CSocket &socket, BYTE pro);
	CIPAddressCtrl ipToD;
	CIPAddressCtrl ipToD2;
	CComboBox combo_c;
	bool ipTOD_in = false;
	CEdit editInfo;
	afx_msg void OnBnClickedButtonDemarcate();
	afx_msg void OnBnClickedButtonFaceDemarcate2();
	afx_msg void OnBnClickedButtonDcabDemarcate3();
	afx_msg void OnBnClickedButtonLeft();
	afx_msg void OnBnClickedButtonRight();
	afx_msg void OnBnClickedButtonDown();
	afx_msg void OnBnClickedButtonAmp();
	afx_msg void OnBnClickedButtonShr();
	afx_msg void OnBnClickedButtonFileSelect();
	CEdit editFilePath;
	afx_msg void OnBnClickedButtonDemarcateEnd();
	afx_msg void OnBnClickedButtonFileSave();
	afx_msg void OnBnClickedCheckLaneDL();
	CEdit editLaneDL;
	CEdit editLaneDR;
	CEdit editLaneRANGE;
	CEdit editSHOUDER_L;
	CEdit editSHOUDER_R;
	CEdit editNOSE_L;
	CEdit editNOSE_R;
	CEdit editNOSE_C;
	CButton cboxLaneLlock;
	CButton cboxLaneRlock;
	afx_msg void OnBnClickedCheckLaneDR();
	int save_config(struct adas_camera* p);
	afx_msg void OnBnClickedButtonFileSend();

	CEdit editHint;
	afx_msg void OnIpnFieldchangedIpaddressIp(NMHDR *pNMHDR, LRESULT *pResult);
	afx_msg void OnIpnFieldchangedIpaddressIp2(NMHDR *pNMHDR, LRESULT *pResult);
	afx_msg void OnEnChangeEditLaneDL();
	afx_msg void OnEnChangeEditLaneRange();
	afx_msg void OnEnChangeEditLaneDR();
	afx_msg void OnEnChangeEditShouderL();
	afx_msg void OnEnChangeShouderR();
	afx_msg void OnEnChangeEditLshouder();
	afx_msg void OnEnChangeEditRshouder();
	afx_msg void OnEnChangeEditLnose();
	afx_msg void OnEnChangeEditRnose();
	afx_msg void OnEnChangeEditHint();
	afx_msg void OnBnClickedLFace();
	afx_msg void OnBnClickedRFace();
	afx_msg void OnBnClickedCanncel();
	afx_msg void OnBnClickedCal();
	afx_msg void OnBnClickedCFace();
	afx_msg void OnCbnSelchangeCombo1();
};
