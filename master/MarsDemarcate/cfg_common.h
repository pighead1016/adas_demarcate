#ifndef _CFG_COMMON_H_  
#define _CFG_COMMON_H_ 

#define PICKUP_FILE_PATH_PREFIX		"/tmp/snap-0.jpg"
#define BACKUP_FILE_PATH_PREFIX		"./outjpg/"
#define PICKUP_ORDER_PREFIX			"./sample-Encoder-jpeg"


typedef struct st_Image_Parameter {

	//unsigned int iISMainThread; //图像识别主线程开关，默认启动打开=1；主监控程序可令其关闭=0，关闭后程序退出。
	//unsigned int iWorkState; //主程序（图像识别）工作状态 1=正在工作；0=休息；默认=1；
		//unsigned int iEQWorkState; //设备当前工作状态 1=正在工作；0=休息；默认=0；
		
		//time_t tWorkStartTime; //当前或下个工作周期时间段开始日期时间
		//time_t tWorkStopTime; //当前或下个工作周期时间段结束日期时间
		//unsigned int size; //提取图片的时间周期，单位：秒，默认=1s，（存储配置文件，实时修改回写，采用工作状态下不间断采集，该参数暂不使用）
		//time_t tLastPickTime; //上次提取图片的日期时间。工作状态取当前时间差值，大于提取图片的时间周期时，执行识别过程。
	//unsigned int iBodyState; //当前传感器识别状态 n(大于0)=有人状态；0=无人状态；默认=0;
	int iN2STimers; //无人状态变为有人状态的条件参数：在无人状态下连续识别出有人iN2STimers次，则改变识别状态iBodyState=1;默认=1（存储配置文件，实时修改回写）
	int iS2NTimers; //有人状态变为无人状态的条件参数：在有人状态下连续识别出无人iS2NTimers次，则改变识别状态iBodyState=0;默认=1（存储配置文件，实时修改回写）
	int iModelSwitch; //更换模板的阈值，
									//iModelSwitchType=0时,经测试静态景物自然光变化前后差值在600000-1000000之间，所以初始值设为800000；当采集图片与模板像素亮度差值到该阈值时，同时采集图片为无人状态，则更换模板。
									//iModelSwitchType=1时,阈值为更新模板间隔秒数。
	int iModelSwitchType;//模板更新类型：0=按照亮度值作为阈值；1=按照时间（秒）做阈值
	int iModelUpdateJudgeTimers;
		//unsigned int iN2STNum; //当前无人状态连续识别出有人次数，默认=0
		//unsigned int iS2NTNum; //当前有人状态连续识别出无人次数，默认=0
		//unsigned int iINIChange; //存储配置文件中的参数发生变化，随即更改该参数为1，以其保存当前配置,默认=0
	int iOutLog; //是否输出日志文件,默认=0不输出
	int iModelDelay; //启动后延时记载模板，阈值为延时间隔秒数。
	//unsigned int iModelNow; //及时更新模板指令。默认=0：没有更新指令，1：有即时更新指令，执行后修改为0
	int iOutImg; //是否输出图像文件,默认=0不输出
	int iNobodyTimers;
	int iRecgMethod;
	int iRecgMethodValue;
	int iThreshold;
	int iDistanceMinCompare;
	int iPointX;
	int iPointY;
	int iWidth;
	int iHighth;
	int iEnforceUpdateTimers;
	int iBoundRectThreshold;
	float iSvmHogRegon;
	int iExit;
} typedef_st_Image_Parameter;





int saveConfig(void *Config_parameters);


int readParameter(const char *file_path,const char *key, char *keyvalue);

int fWriteLog(char *sContent);
int getNowTime(char *nowTime);
int getFileTime(char *FileTime);
int SaveFileTime();


//#define _DEBUG
#ifdef _DEBUG
#define DEBUG(format,...) printf("FILE: "__FILE__", LINE: %d: "format"/n", __LINE__, ##__VA_ARGS__)
#else
#define DEBUG(format,...)
#endif

extern typedef_st_Image_Parameter global_image_para;

#endif


