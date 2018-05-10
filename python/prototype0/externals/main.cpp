
#include <Windows.h>
#include <sstream>
#include <iostream>
#include <math.h> 
#include "HackProcess.h"
#include <vector>
#include <algorithm>  
#include <cstring>
#include <SFML\Graphics.hpp>
#include <SFML\Network.hpp>
#include <chrono>
#include <thread>
#include <string>

//char sizeChr[] = { size & 0x000000ff, (size & 0x0000ff00) >> 8 , (size & 0x00ff0000) >> 16, (size & 0xff000000) >> 24 };
#define INT2CHARarr(size){ size & 0x000000ff, (size & 0x0000ff00) >> 8 , (size & 0x00ff0000) >> 16, (size & 0xff000000) >> 24 }

//----------------AIMBOT RELATED CODE------------------
//Create our 'hooking' and process managing object
CHackProcess fProcess;
using namespace std;
using namespace std::chrono;

#define F6_Key 0x75
#define RIGHT_MOUSE 0x02
int NumOfPlayers = 64;
int NumEnemy = 32;
const DWORD dw_PlayerCountOffs = 0x5D39BC;//Engine.dll
const DWORD LocalPlayer = 0x4C6708;//0x00574560;
const DWORD dw_mTeamOffset = 0x9C;//client
const DWORD dw_Health = 0x94;//client
const DWORD dw_Pos = 0x260;//client
const DWORD dw_Rot = 0x26C;//client    //engine.dll+4632D4 if wanna write or read from engine.dll
const DWORD EntityList = 0x4D3904;
const DWORD EntityLoopDistance = 0x10;
const DWORD dw_vMatrix_1 = 0x597EF0;
const DWORD dw_CrosshairIdOffset = 0x14F0;

//ESP VARS
const DWORD dw_vMatrix_2 = 0xE6D00168;
const DWORD dw_vMatrix_3 = 0xE6D00478;

float m_line[4];




//ViewAngles
//We find these by moving our mouse around constantly looking for changed/unchanged value,
//the alternative is to use cl_pdump 1 and search for the value assigned to m_angRotation vector
//const DWORD dw_m_angRotation = 0x461A9C;
RECT m_Rect;

//Set of initial variables you'll need
//Our desktop handle
HDC HDC_Desktop;
//Brush to paint ESP etc
HBRUSH EnemyBrush;
HFONT Font; //font we use to write text with


HWND TargetWnd;
HWND Handle;
DWORD DwProcId;

COLORREF SnapLineCOLOR;
COLORREF TextCOLOR;


class Timer {
private:
	unsigned long begTime;
public:
	void start() {
		begTime = clock();
	}
	unsigned long elapsedTime() {
		return ((unsigned long)clock() - begTime);
	}
	bool isTimeout(unsigned long seconds) {
		return seconds >= elapsedTime();
	}
};

typedef struct
{
	float flMatrix[4][4];
}WorldToScreenMatrix_t;


float Get3dDistance(float * myCoords, float * enemyCoords)
{
	return sqrt(
		pow(double(enemyCoords[0] - myCoords[0]), 2.0) +
		pow(double(enemyCoords[1] - myCoords[1]), 2.0) +
		pow(double(enemyCoords[2] - myCoords[2]), 2.0));

}


void SetupDrawing(HDC hDesktop, HWND handle)
{
	HDC_Desktop = hDesktop;
	Handle = handle;
	EnemyBrush = CreateSolidBrush(RGB(255, 0, 0));
	//Color
	SnapLineCOLOR = RGB(0, 0, 255);
	TextCOLOR = RGB(0, 255, 0);
}

//We will use this struct throughout all other tutorials adding more variables every time
struct MyPlayer_t
{
	DWORD CLocalPlayer;
	int Team;
	int Health;
	WorldToScreenMatrix_t WorldToScreenMatrix;
	float Position[3];
	int flickerCheck;
	int CrossHairEntityId;
	float AimbotAngle[3];
	void ReadInformation()
	{
		// Reading CLocalPlayer Pointer to our "CLocalPlayer" DWORD.
		ReadProcessMemory(fProcess.__HandleProcess, (PBYTE*)(fProcess.__dwordClient + LocalPlayer), &CLocalPlayer, sizeof(DWORD), 0);
		// Reading out our Team to our "Team" Varible.
		ReadProcessMemory(fProcess.__HandleProcess, (PBYTE*)(CLocalPlayer + dw_mTeamOffset), &Team, sizeof(int), 0);
		// Reading out our Health to our "Health" Varible.    
		ReadProcessMemory(fProcess.__HandleProcess, (PBYTE*)(CLocalPlayer + dw_Health), &Health, sizeof(int), 0);
		// Reading out our Position to our "Position" Varible.
		ReadProcessMemory(fProcess.__HandleProcess, (PBYTE*)(CLocalPlayer + dw_Pos), &Position, sizeof(float[3]), 0);
		// Current thing on the crosshair
		ReadProcessMemory(fProcess.__HandleProcess, (PBYTE*)(CLocalPlayer + dw_CrosshairIdOffset), &CrossHairEntityId, sizeof(int), 0);
		// Reading angle of the player
		ReadProcessMemory(fProcess.__HandleProcess, (PBYTE*)(CLocalPlayer + dw_Rot), &AimbotAngle, sizeof(float[3]), 0);

		//Here we find how many player entities exist in our game, through this we make sure to only loop the amount of times we need
		//when grabbing player data
		//Note that this call could be even better at a regular 15 or so seconds timer but performance shouldn't vary a great deal
		ReadProcessMemory(fProcess.__HandleProcess, (PBYTE*)(fProcess.__dwordEngine + dw_PlayerCountOffs), &NumOfPlayers, sizeof(int), 0);


		//WriteProcessMemory(fProcess.__HandleProcess,(PBYTE*)(fProcess.__dwordEngine + dw_m_angRotation),TargetList[0].AimbotAngle, 12, 0);
		
		//anti flicker
		//VMATRIX
		if (flickerCheck == 0)
		{
			ReadProcessMemory(fProcess.__HandleProcess, (PBYTE*)(fProcess.__dwordEngine + dw_vMatrix_1), &WorldToScreenMatrix, sizeof(WorldToScreenMatrix), 0);
		}
		//-1A4 = ANTI FLICKER
		//Engine.dll+0x58C45C
	}
}MyPlayer;


struct TargetList_t
{
	float Distance;
	float AimbotAngle[3];

	TargetList_t()
	{
	}

	TargetList_t(float aimbotAngle[], float myCoords[], float enemyCoords[])
	{
		//Send our coordinates and the enemy's to find out how close they are to us!
		Distance = Get3dDistance(myCoords[0], myCoords[1], myCoords[2],
			enemyCoords[0], enemyCoords[1], enemyCoords[2]);

		//Define our aimbot angles and set them for later use when shooting
		AimbotAngle[0] = aimbotAngle[0];
		AimbotAngle[1] = aimbotAngle[1];
		AimbotAngle[2] = aimbotAngle[2];
	}

	//Get our 3d Distance between 2 sets of coordinates(ours and enemies) and find out how close an enemy is to us
	//when it comes to shooting we aim at the closest enemy
	//Simple but effective

	float Get3dDistance(float myCoordsX, float myCoordsZ, float myCoordsY,
		float eNx, float eNz, float eNy)
	{
		return sqrt(
			pow(double(eNx - myCoordsX), 2.0) +
			pow(double(eNy - myCoordsY), 2.0) +
			pow(double(eNz - myCoordsZ), 2.0));
	}
};




void CalcAngle(float *src, float *dst, float *angles)
{
	double delta[3] = { (src[0] - dst[0]), (src[1] - dst[1]), (src[2] - dst[2]) };
	double hyp = sqrt(delta[0] * delta[0] + delta[1] * delta[1]);
	angles[0] = (float)(asinf(delta[2] / hyp) * 57.295779513082f);
	angles[1] = (float)(atanf(delta[1] / delta[0]) * 57.295779513082f);
	angles[2] = 0.0f;

	if (delta[0] >= 0.0)
	{
		angles[1] += 180.0f;
	}
}




//ENemy struct
struct PlayerList_t
{
	DWORD CBaseEntity;
	int Team;
	int Health;
	float Position[3];
	float AimbotAngle[3];

	void ReadInformation(int Player)
	{
		// Reading CBaseEntity Pointer to our "CBaseEntity" DWORD + Current Player in the loop. 0x10 is the CBaseEntity List Size
		//"client.dll"+00545204 //0x571A5204
		ReadProcessMemory(fProcess.__HandleProcess, (PBYTE*)(fProcess.__dwordClient + EntityList + (Player * EntityLoopDistance)), &CBaseEntity, sizeof(DWORD), 0);
		// Reading out our Team to our "Team" Varible.
		ReadProcessMemory(fProcess.__HandleProcess, (PBYTE*)(CBaseEntity + dw_mTeamOffset), &Team, sizeof(int), 0);
		// Reading out our Health to our "Health" Varible.    
		ReadProcessMemory(fProcess.__HandleProcess, (PBYTE*)(CBaseEntity + dw_Health), &Health, sizeof(int), 0);
		// Reading out our Position to our "Position" Varible.
		ReadProcessMemory(fProcess.__HandleProcess, (PBYTE*)(CBaseEntity + dw_Pos), &Position, sizeof(float[3]), 0);
	}
}PlayerList[32];


bool WorldToScreen(float * from, float * to)
{
	float w = 0.0f;

	to[0] = MyPlayer.WorldToScreenMatrix.flMatrix[0][0] * from[0] + MyPlayer.WorldToScreenMatrix.flMatrix[0][1] * from[1] + MyPlayer.WorldToScreenMatrix.flMatrix[0][2] * from[2] + MyPlayer.WorldToScreenMatrix.flMatrix[0][3];
	to[1] = MyPlayer.WorldToScreenMatrix.flMatrix[1][0] * from[0] + MyPlayer.WorldToScreenMatrix.flMatrix[1][1] * from[1] + MyPlayer.WorldToScreenMatrix.flMatrix[1][2] * from[2] + MyPlayer.WorldToScreenMatrix.flMatrix[1][3];
	w = MyPlayer.WorldToScreenMatrix.flMatrix[3][0] * from[0] + MyPlayer.WorldToScreenMatrix.flMatrix[3][1] * from[1] + MyPlayer.WorldToScreenMatrix.flMatrix[3][2] * from[2] + MyPlayer.WorldToScreenMatrix.flMatrix[3][3];

	if (w < 0.01f)
		return false;

	float invw = 1.0f / w;
	to[0] *= invw;
	to[1] *= invw;

	int width = (int)(m_Rect.right - m_Rect.left);
	int height = (int)(m_Rect.bottom - m_Rect.top);

	float x = width / 2;
	float y = height / 2;

	x += 0.5 * to[0] * width + 0.5;
	y -= 0.5 * to[1] * height + 0.5;

	to[0] = x + m_Rect.left;
	to[1] = y + m_Rect.top;

	return true;
}


struct vec4 {
	int val[4];
	vec4(int a, int b, int c, int d) {
		val[0] = a; val[1] = b; val[2] = c; val[3] = d;
	}
};
struct vec5 {
	int val[5];
	vec5(int a, int b, int c, int d, int e) {
		val[0] = a; val[1] = b; val[2] = c; val[3] = d; val[4] = e;
	}
};

vector<vec5> border;

void DrawESP2(int x, int y, int x2, int y2, float distance, int playerIndex)
{
	y2 = y2 - 15;
	x2 = x2 - 2;
	y = y - 15;
	x = x - 2;

	border.push_back(vec5(x2 - (y - y2) / 4, y2, (int)distance, y - y2, playerIndex));

	//x1,y1, width, height ///here width = height/2
}

void ESP()
{
	GetWindowRect(FindWindowA(NULL, "Counter-Strike Source"), &m_Rect);

	for (int i = 0; i < NumOfPlayers; i++)
	{
		PlayerList[i].ReadInformation(i);

		if (PlayerList[i].Health < 2)
			continue;

		if (PlayerList[i].Team == MyPlayer.Team)
			continue;


		if (PlayerList[i].Team == MyPlayer.Team)
			continue;
		float EnemyXY[3];
		float EnemyXY2[3];

		PlayerList[i].Position[2] += 10; //0;
		if (WorldToScreen(PlayerList[i].Position, EnemyXY))
		{
			//DrawESP(EnemyXY[0] - m_Rect.left, EnemyXY[1] - m_Rect.top, Get3dDistance(MyPlayer.Position, PlayerList[i].Position));
			PlayerList[i].Position[2] += 60; //64;
			WorldToScreen(PlayerList[i].Position, EnemyXY2);

			DrawESP2(EnemyXY[0] - m_Rect.left, EnemyXY[1] - m_Rect.top, EnemyXY2[0] - m_Rect.left, EnemyXY2[1] - m_Rect.top, Get3dDistance(MyPlayer.Position, PlayerList[i].Position), i);
		}

	}




}

vector<TargetList_t> TargetList;

void CalculateTargets()
{
	TargetList.clear();
	NumEnemy = 0;
	for (int i = 0; i < NumOfPlayers; i++)
	{
		PlayerList[i].ReadInformation(i);

		// Skip if they're my teammates. 
		if (PlayerList[i].Team == MyPlayer.Team) {
			continue;
		}
		else NumEnemy++;
		if (PlayerList[i].Health < 2) {
			continue;
		} //break to next iteration
		  //PlayerList[i].Position[2] -= 10;
		CalcAngle(MyPlayer.Position, PlayerList[i].Position, PlayerList[i].AimbotAngle);

		TargetList.push_back(TargetList_t(PlayerList[i].AimbotAngle, MyPlayer.Position, PlayerList[i].Position));
	}
}

void DisplayPlayersInfo() {

	for (int a = 0; a < NumOfPlayers; a++) {
		
		if (a == 0 && PlayerList[a].CBaseEntity<34)
			cout << PlayerList[a].CBaseEntity - 1;
		else
			cout << a;
		cout << "\t";
		cout << PlayerList[a].Health;
		cout << "\t";
		cout << PlayerList[a].Team;
		cout << "\t";
		//cout << std::cout.precision(2) << std::fixed;
		cout << PlayerList[a].Position[0];
		cout << "\t";
		cout << PlayerList[a].Position[1];
		cout << "\t";
		cout << PlayerList[a].Position[2];
		cout << "\t";
		cout << PlayerList[a].AimbotAngle[0];
		cout << "\t";
		cout << PlayerList[a].AimbotAngle[1];
		cout << "\t";
		cout << PlayerList[a].AimbotAngle[2];
		cout << endl;
	}

}

struct CompareDistance
{
	//USE A COMPARATOR TO SORT OUR ARRAY nicely
	bool operator() (vec5 & lhs, vec5 & rhs)
	{
		return lhs.val[2] < rhs.val[2];
	}
};

vector<char> socketData;
int main()
{
	fProcess.RunProcess();

	//ShowWindow(FindWindow("ConsoleWindowClass", NULL), false);
	TargetWnd = FindWindow(0, "Counter-Strike Source");


	sf::TcpSocket socket;
	sf::IpAddress ip = sf::IpAddress::LocalHost;

	char buffer[1024];
	size_t received;
	string text = "connected to c++ Server";
	cout << "the ip is " << ip.LocalHost << endl;

	sf::TcpListener listener;
	listener.listen(2000);
	listener.accept(socket);

	cout << buffer << endl;

	Timer timer;
	int fps = 0;
	timer.start();

	while (true)
	{

		MyPlayer.ReadInformation();
		CalculateTargets();
		ESP();
		socketData.clear();
		sort(border.begin(), border.end(), CompareDistance());


		for (int a = 0; a < border.size(); a++)
			border[a].val[2] = border[a].val[3] / 2;

		//crosshair entity as cbase for my player
		PlayerList[0].CBaseEntity = (unsigned long)MyPlayer.CrossHairEntityId;
		//myplayer aimbot as its current angle 
		PlayerList[0].AimbotAngle[0] = MyPlayer.AimbotAngle[0];
		PlayerList[0].AimbotAngle[1] = MyPlayer.AimbotAngle[1];
		PlayerList[0].AimbotAngle[2] = MyPlayer.AimbotAngle[2];

		//send begining of data information
		char beginIndicator[] = { '7' };
		socket.send(beginIndicator, sizeof(beginIndicator));
		Sleep(1);


		string datas = "";

		{
			// how many borders/box of enemy
			int size = border.size();
			char sizeChr[] = INT2CHARarr(size);
			for (int no = 0; no < sizeof(sizeChr); no++)
				socketData.push_back(sizeChr[no]);
		} {
			// how many information of teamPlayers
			//int size = NumOfPlayers - NumEnemy;
			int size = NumOfPlayers;
			char sizeChr[] = INT2CHARarr(size);
			for (int no = 0; no < sizeof(sizeChr); no++)
				socketData.push_back(sizeChr[no]);
		}

		//sending size of information 
		datas = std::string(socketData.begin(), socketData.end());
		socket.send(datas.c_str(), datas.size());
		socketData.clear();
		datas.clear();

		Sleep(1);

		//box information
		{
			char sendData[sizeof(int[5])];
			for (int i = 0; i < border.size(); i++) {
				int tempInt[5] = { border[i].val[0], border[i].val[1], border[i].val[2], border[i].val[3], border[i].val[4] };
				memcpy(sendData, tempInt, sizeof(sendData));
				for (int no = 0; no < sizeof(sendData); no++)
					socketData.push_back(sendData[no]);
			}
		}
		//player information
		{
			char sendData[sizeof(PlayerList_t)];
			for (int i = 0; i < NumOfPlayers; i++) {
				//if (PlayerList[i].Team != MyPlayer.Team) continue;
				memcpy(sendData, &PlayerList[i], sizeof(sendData));
				for (int no = 0; no < sizeof(sendData); no++)
					socketData.push_back(sendData[no]);
			}
		}

		//cout <<data<< data.size()<<endl;
		datas = std::string(socketData.begin(), socketData.end());
		//int a = int((datas[0]) << 24 |
		//	(datas[1]) << 16 |
		//	(datas[2]) << 8 |
		//	(datas[3]));
		//cout << a << " == "<<border[0].val[0]<<" == "<<datas.substr(0,4)<<endl;
		socket.send(datas.c_str(), datas.size());












		border.clear();
		fps++;
		if (timer.elapsedTime() > 999l) {
			system("cls");
			DisplayPlayersInfo();
			cout << "Dps = " << fps << endl;
			//cout << "Target = "<< MyPlayer.CrossHairEntityId<<endl;
			timer.start();
			fps = 0;
		}


		Sleep(10);
	}


	return 0;
}
