#!/usr/bin/Rscript

require(MBatch)
suppressPackageStartupMessages(library(dplyr))
suppressPackageStartupMessages(library(readr))
suppressPackageStartupMessages(library(stringr))

#######
# Paths must be based on Docker container paths
inputDir <- '/bk_mbatch/data/inputs'
outputDir <- '/bk_mbatch/data/mbatch'
run='v05'

# theDataFile1=cleanFilePath(inputDir, "cptac_colon_adeno_transcriptomics_ref.tsv") # ref
# theDataFile2=cleanFilePath(inputDir, "cell_line_colon_adeno_transcriptomics_src.tsv") # data to be projected
theDataFile1=cleanFilePath(inputDir, "cptac_glioblastoma_transcriptomics_ref.tsv") # ref
theDataFile2=cleanFilePath(inputDir, "cell_line_glioblastoma_transcriptomics_src.tsv") # data to be projected

theOutputDir=cleanFilePath(outputDir, "EBNPlus_TrainAndValidateReplicates_Structures")
theBatchId1="ref"
theBatchId2="src"
theRandomSeed=314
#######

set.seed(theRandomSeed)

# remove duplicates from columns (samples)
removeDuplicatesFromColumns <- function(theMatrix)
{
  indexOfDuplicates <- which(duplicated(colnames(theMatrix)))
  if (length(indexOfDuplicates) > 0)
  {
    # minus sign uses inverse of indexes
    theMatrix <- theMatrix[ ,-indexOfDuplicates]
  }
  return(theMatrix)
}

# remove duplicates from rows (genes/probes)
removeDuplicatesFromRows <- function(theMatrix)
{
  indexOfDuplicates <- which(duplicated(rownames(theMatrix)))
  if (length(indexOfDuplicates) > 0)
  {
    # minus sign uses inverse of indexes
    theMatrix <- theMatrix[-indexOfDuplicates, ]
  }
  return(theMatrix)
}

printMatrix <- function(theMatrix)
{
  print(is.matrix(theMatrix))
  print(dim(theMatrix))
  rowMax <- dim(theMatrix)[1]
  colMax <- dim(theMatrix)[2]
  rowMax <- min(rowMax, 4)
  colMax <- min(colMax, 4)
  print(theMatrix[1:rowMax, 1:colMax])
}


warnLevel<-getOption("warn")
on.exit(options(warn=warnLevel))
# warnings are errors
options(warn=3)
# if there is a warning, show the calls leading up to it
options(showWarnCalls=TRUE)
# if there is an error, show the calls leading up to it
options(showErrorCalls=TRUE)
#
unlink(theOutputDir, recursive=TRUE)
dir.create(theOutputDir, showWarnings=FALSE, recursive=TRUE)

# read the files in. This can be done however you want
print("read the files")
theDataMatrix1 <- readAsGenericMatrix(theDataFile1)
theDataMatrix2 <- readAsGenericMatrix(theDataFile2)

# remove any duplicates (this is a requirement for EBNplus)
print("remove duplicates")
theDataMatrix1 <- removeDuplicatesFromColumns(removeDuplicatesFromRows(theDataMatrix1))
theDataMatrix2 <- removeDuplicatesFromColumns(removeDuplicatesFromRows(theDataMatrix2))


# Log2 normalize data (post MBatch, will need to apply the antilog)
theDataMatrix1 <- log2(theDataMatrix1) %>% as.matrix()
theDataMatrix2 <- log2(theDataMatrix2) %>% as.matrix()


# Set up params for MBatch EBN+
# use all samples
theEBNP_PsuedoReplicates1Train <- c(colnames(theDataMatrix1))
theEBNP_PsuedoReplicates2Train <- c(colnames(theDataMatrix2))


# Run MBatch EBN+ function
print("EBNPlus_TrainAndValidateReplicates_Structures")
theEBNP_PsuedoReplicates1Validation <- NULL # set to NULL if transforming a data into data space of other data
theEBNP_PsuedoReplicates2Validation <- NULL # set to NULL if transforming a data into data space of other data
resultsList <- EBNPlus_TrainAndValidateFromVector_Structures(theDataMatrix1, theDataMatrix2,
      theBatchId1, theBatchId2,
      theEBNP_PsuedoReplicates1Train,
      theEBNP_PsuedoReplicates2Train,
      theEBNP_PsuedoReplicates1Validation,
      theEBNP_PsuedoReplicates2Validation,
      theEBNP_BatchWithZero="both",
      theEBNP_FixDataSet=1,
      theEBNP_CorrectForZero=TRUE,
      theEBNP_ParametricPriorsFlag=TRUE,
      theEBNP_TestRatio=0,
      theSeed=theRandomSeed,
      theTestSeed=theRandomSeed,
      theEBNP_PriorPlotsFile=cleanFilePath(theOutputDir, paste(run, "priorplots.PNG",sep='_')))

# # If running function to correct a dataset,
# # these will not be fully populated:
# # resultsList$TestSet1 /2, ValidationSet1 /2, resultsList$ValidationResults
print("TrainingSet1")
printMatrix(resultsList$TrainingSet1)
print("TrainingSet2")
printMatrix(resultsList$TrainingSet2)
print("TrainingResults")
printMatrix(resultsList$TrainingResults)
print("CorrectedResults")
printMatrix(resultsList$CorrectedResults)


# Transform back to linear space (antilog2)
print('apply antilog to transform back to linear space')
resultsList$CorrectedResults <- 2^resultsList$CorrectedResults
theDataMatrix1 <- 2^theDataMatrix1
theDataMatrix2 <- 2^theDataMatrix2

# Save output
print("Writing corrected matrix output file (contains all data)")
write.table(resultsList$CorrectedResults, paste(outputDir, '/correctedResults_', theBatchId2, '_', run, '.tsv', sep=''), sep= "\t", row.names=TRUE, col.names=TRUE)
print("Writing full input file 1 (static)")
write.table(theDataMatrix1, paste(outputDir, '/input_', theBatchId1, '_', run, '.tsv', sep=''), sep= "\t", row.names=TRUE, col.names=TRUE)
print("Writing full input file 2 (corrected)")
write.table(theDataMatrix2, paste(outputDir, '/input_', theBatchId2, '_', run, '.tsv', sep=''), sep= "\t", row.names=TRUE, col.names=TRUE)
