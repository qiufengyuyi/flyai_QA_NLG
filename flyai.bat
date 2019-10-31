
@rem ----- ExeScript Options Begin -----
                                
                           
               
                                      
                                   
                         
@rem ----- ExeScript Options End -----

@echo off

if not exist C:\%HOMEPATH%\.flyai (
   md C:\%HOMEPATH%\.flyai
)
if not exist C:\%HOMEPATH%\.flyai\flyai_check.exe (
	bitsadmin.exe /transfer "FlyAI" https://dataset.flyai.com/flyai_check_windows_v0.3.exe C:\%HOMEPATH%\.flyai\flyai_check.exe
	bitsadmin.exe /setpriority "FlyAI" foreground
)
C:\%HOMEPATH%\.flyai\flyai_check.exe
C:\%HOMEPATH%\.flyai\flyai.exe %*
