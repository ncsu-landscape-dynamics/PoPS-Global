require(rgdal)
require(plyr)
require(raster)

un_list <- read.csv('H:\\Shared drives\\APHIS  Projects\\Pandemic\\Data\\Comtrade\\UN_Comtrade_CountryList.csv')
kgcodes <- read.csv('H:\\Shared drives\\Data\\Raster\\Global\\Beck_KoppenClimate\\KGcodes.csv', stringsAsFactors = F)
borders <- readOGR('H:\\Shared drives\\APHIS  Projects\\Pandemic\\Data\\Country_list_shapefile\\TM_WORLD_BORDERS-0.3\\TM_WORLD_BORDERS-0.3.shp')

#kgc50 <- raster('H:\\Shared drives\\Data\\Raster\\Global\\Beck_KoppenClimate\\Beck_KG_V1_present_0p5.tif')
kgc10 <- raster('H:\\Shared drives\\Data\\Raster\\Global\\Beck_KoppenClimate\\Beck_KG_V1_present_0p083.tif')
#kgc01 <- raster('H:\\Shared drives\\Data\\Raster\\Global\\Beck_KoppenClimate\\Beck_KG_V1_present_0p0083.tif')

kg.df <- ldply(.data=c(1:length(borders$NAME)), 
               .fun=function(X){
                 X.ct <- borders[X,]
                 X.ex <- extract(x=kgc10, y=X.ct, method='bilinear')[[1]]
                 X.n <- length((X.ex)[intersect(which((!is.na(X.ex))), which((X.ex)>0))])
                 X.df <- t(data.frame(rep(0, nrow(kgcodes))))
                 colnames(X.df) <- kgcodes$let
                 row.names(X.df) <- X.ct$NAME
                 
                 for(i in 1:ncol(X.df)){
                   X.df[, i] <- sum((X.ex)==kgcodes$num[i], na.rm=T)/X.n
                 }
                 
                 return(X.df)
               }, .progress='text')
row.names(kg.df) <- borders$NAME