# This script was used to generate the simulated Splatter data-sets for the project. The input parameters may be not entirely up to date with the parameter values used in the actual experiments performed for the thesis.

# Import libraries:
install.packages('BiocManager')
library('BiocManager')
BiocManager::install('scater')
library(scater)
BiocManager::install('splatter')
library('splatter')

# Generate new SplatParams object. This is only necessary once.
params <- newSplatParams()

# Loop over number of genes:
for (i in c(5,25,50,150,250, 500)) {
  
  # Setting all parameter values
  params <- setParam(params, 'nGenes', i)
    
  # Small data-set
  params<- setParam(params, 'group.prob',c(0.2,0.2,0.2,0.2,0.2))
  de_prob = 2.5/i
  params <- setParam(params, 'de.prob',c(de_prob,de_prob,de_prob,de_prob,de_prob))
  params<- setParam(params, 'de.facLoc',3)
  params <- setParam(params, 'batchCells', 500)
  params<- setParam(params, 'de.facScale',0)
  
  # Simulate and extract data
  sim <- splatSimulateGroups(params)
  df <- counts(sim)
 
  # Write data
  write.csv(df,paste0("./splatter_small_",i,".csv"))
  
  # Extract and write labels
  labels <- colData(sim)$Group
  write.csv(labels,paste0("./splatter_small_",i,"_labels.csv"))
    
  # Big data-set
  params<- setParam(params, 'group.prob',c(0.24,0.12,0.10,0.02,0.37,0.15))
  de_prob = 1/(i/80)
  params <- setParam(params, 'de.prob',c(de_prob,de_prob,de_prob,de_prob,de_prob,de_prob))
  params<- setParam(params, 'de.facLoc',0.1)
  params <- setParam(params, 'batchCells', 750)
  params<- setParam(params, 'de.facScale',0.4)
  
  # Simulate and extract data
  sim <- splatSimulateGroups(params)
  df <- counts(sim)
    
  # Write data
  write.csv(df,paste0("./splatter_big_",i,".csv"))
  
  # Extract and write labels
  labels <- colData(sim)$Group
  write.csv(labels,paste0("./splatter_big_",i,"_labels.csv"))
    
  print(c(i,' done'))
}