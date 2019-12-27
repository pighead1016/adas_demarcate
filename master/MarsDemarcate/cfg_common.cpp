
#include "stdafx.h"
#include "cfg_common.h"

#define BUFFER_MAX 100
#define DIRECTION_MAX 100
#define IN 0
#define OUT 1
#define LOW 0
#define HIGH 1
#define POUT 49


/* 从字符串的左边截取n个字符 */
int strLeft(char *dst,char *src, long n)  
{  
    char *p = src;  
    char *q = dst;  
    int len = strlen(src);  
    if(n>len) n = len;  
    while(n--) *(q++) = *(p++);  
    *(q++)='\0';
    return 0;  
}
int strRight(char *dst,char *src, int n)  
{  
    char *p = src;  
    char *q = dst;  
    int len = strlen(src);  
    if(n>len) n = len;  
    p += (len-n);   //从右边第n个字符开始，到0结束  
    while((*(q++) = *(p++)));  
    return 0;  
} 


int readParameter(const char *file_path,const char *key, char *keyvalue)
{

    int ERRO_MSG = 1;
	char mykeys[100]={0};
	
	strcat_s(mykeys,key);
	strcat_s(mykeys,"=");
	
    //定义文件指针
	FILE *fpr = NULL;
    //打开文件
    fopen_s(&fpr,file_path, "r");
    if (fpr == NULL)
    {
        ERRO_MSG = 2;
        printf("open file erro msg:%d\n", ERRO_MSG);
        return ERRO_MSG;
    }
	
    while (!feof(fpr)){
        char buf[100] = { 0 };
        fgets(buf, 100, fpr);
		if (buf[0]!='#')
		{ 
			char tmpbuf[100]={0}; 
			strLeft(tmpbuf,buf,strlen(mykeys));
			if (strcmp(tmpbuf,mykeys) == 0)
			{
				strRight(tmpbuf,buf,strlen(buf)-strlen(mykeys));
				strLeft(keyvalue,tmpbuf,strlen(tmpbuf)-1);
				ERRO_MSG = 0;
				break;
			}
		}       
    }
    //关闭文件指针
    if (fpr != NULL)
    {
        fclose(fpr);
    }
    return ERRO_MSG;
}