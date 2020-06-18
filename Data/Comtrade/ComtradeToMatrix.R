library(data.table)

filelist = list.files(pattern="*.csv$")





    countries <-read.csv("country.list.csv", header = T, stringsAsFactors = F)
    countries <- countries$Name_UN
    
    
   
    
  for(file in 1:length(filelist)  ){
    

    data <- fread(filelist[file], header = T, stringsAsFactors = F)
    years <- unique(data$year)
    years <- sort(years)
    for(yearind in 1:length(years)){
      
      message(years[yearind])
      matrix <- matrix(0, nrow = length(countries), ncol = length(countries))
      rownames(matrix) <- countries
      colnames(matrix) <- countries
    
      yeardata <- data[year == years[yearind]] 
      
      
      for (rowind in 1: nrow(yeardata)){
        
        
        
        exporter <- yeardata$partner[rowind]

        
        
        
        importer <- yeardata$reporter[rowind]
        
        if( isTRUE(length(exporter) > 0) & exporter %in% countries & importer %in% countries){
        
        

        
        matrix[importer,exporter] <- yeardata$netweight_kg[rowind] }
      }
       
    write.csv(matrix,paste0("CitrusMatrix",years[yearind],".csv"))  # < CHANGE NAMES

    }
  }
 
