








#######################
#   
#Scraping for annual citrus comtrade data 
#  to do: 1. install packages githubinstall, rjson, ISOcodes
#         2. change date range to those assigned to you by email (marked below)
#         3. create new folder and set as WD
#         4. Run this script
#         5. Upload the csv files in your folder to the citrus_annual folder in data on drive
#
#         feel free to set the path of the write.csv statement to the shared drive if thats your workflow
#         expected runtime: 12ish hours each with 3 folks, 18 with 2




######################
library(rjson)
library(githubinstall)
library(ISOcodes)
#githubinstall("comtradr")
library(comtradr)





countries <-read.csv("country.list.csv", header = T, stringsAsFactors = F)
countries <- countries$Name_UN


startYear <- 1990   ###### <- CHANGE ME
endYear <- 2019   ###### <- CHANGE ME

ct_register_token("jXIKwJ2httdcPDHwwJCj7GzbDh8fva23HYV17lyN+BeKrxX3fSviSAT9vgH5zQ+XnKj75SBnqPn25kXrwD1viUgtdDMNhpjrw4ZPcpdznaYq1nH8F/wxSoUBSMUzwVVb3YsoqruN04qDiJU/NleTCA==")
ct_get_remaining_hourly_queries()


yearsCompleteTo <- startYear -1 



while(yearsCompleteTo < endYear){

  queryStartYear <- yearsCompleteTo + 1
  queryEndYear <- queryStartYear
  
  if(queryEndYear > endYear){
    queryEndYear <- endYear
  }
  
 
  
  message(paste0("You are currently on year", queryStartYear))
  for (country in 1:length(countries)) {
    
    
    countryExists <- ct_country_lookup(countries[country], 'reporter', ignore.case = T)
    
    
    if(ct_get_remaining_hourly_queries() < 2){
      message("COMTRADE has says your're in time out")
      Sys.sleep(3600)
    }
      
      
      

    if (countryExists != "No matching results found"){
      
      
      
        for(monthset in 1:3){
          
          if(monthset == 1){
            
            queryStart = paste0(queryStartYear, "-01")
            queryEnd = paste0(queryEndYear, "-04")
          }
          
          else if(monthset == 2){
            
            queryStart = paste0(queryStartYear, "-05")
            queryEnd = paste0(queryEndYear, "-08")
          }
          else {
            
            queryStart = paste0(queryStartYear, "-09")
            queryEnd = paste0(queryEndYear, "-12")
          }
          
          
          
          
    query <- ct_search(countries[country],
                       "All", 
                       trade_direction = "imports",
                       freq = "monthly",  #< CHANGE ME
                       start_date = queryStart,
                       end_date = queryEnd,
                       commod_codes = "0805",  # < CHANGE ME
                       max_rec = 50000)
    
    
    
    
    
    
    
    
    

    
    
    if(!exists("masterData")){
      masterData <- query
      }
    else{
      masterData <- rbind(masterData, query)
      }
      
        }
    }
  }
  
  
  
  
  write.csv(masterData, paste0("citrusAnnual", queryStartYear, ".csv"))
  rm(masterData)
  yearsCompleteTo <- queryEnd
}  


  