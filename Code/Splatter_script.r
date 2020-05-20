install.packages('BiocManager')
library('BiocManager')
BiocManager::install('scater')
library(scater)
BiocManager::install('splatter')
library('splatter')

for (i in c(5,25,50,150,250, 500)) {
  params <- newSplatParams()
  params <- setParam(params, 'nGenes', i)
    
  #small
  #params<- setParam(params, 'group.prob',c(0.2,0.2,0.2,0.2,0.2))
  #de_prob = 2.5/i
  #params <- setParam(params, 'de.prob',c(de_prob,de_prob,de_prob,de_prob,de_prob))
  #params<- setParam(params, 'de.facLoc',3)
  #params <- setParam(params, 'batchCells', 500)
  #params<- setParam(params, 'de.facScale',0)
    
  #big
  params<- setParam(params, 'group.prob',c(0.24,0.12,0.10,0.02,0.37,0.15))
  de_prob = 2.5/i
  params <- setParam(params, 'de.prob',c(de_prob,de_prob,de_prob,de_prob,de_prob,de_prob))
  params<- setParam(params, 'de.facLoc',0.1)
  params <- setParam(params, 'batchCells', 750)
  params<- setParam(params, 'de.facScale',0.4)
  
  sim <- splatSimulateGroups(params)
  df <- counts(sim)
    
  #write.csv(df,paste0("./splatter_small_",i,".csv"))
  write.csv(df,paste0("./splatter_big_",i,".csv"))
    
  labels <- colData(sim)$Group
  #write.csv(labels,paste0("./splatter_small_",i,"_labels.csv"))
  write.csv(labels,paste0("./splatter_big_",i,"_labels.csv"))
    
  print(c(i,' done'))
}
