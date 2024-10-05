library(geomorph)
M <- gpagen(plethodon$land, ProcD=TRUE, Proj=FALSE, PrinAxes=FALSE)

# Examine
plotAllSpecimens(M$coords, links=plethodon$links)
max(abs(apply(M$coords, c(1,2), mean) - M$consensus))

# Reshape for use with my Python code.
p <- dim(plethodon$land)[1]
k <- dim(plethodon$land)[2]
n <- dim(plethodon$land)[3]
input_shapes <- matrix(0, nrow=n, ncol=p*k)
align_shapes <- matrix(0, nrow=n, ncol=p*k)

for (i in 1:n) {
	N <- plethodon$land[,,i]
	P <- t(N)
	dim(P) <- c(1, 24)	
	input_shapes[i,] <- P
	
	N <- M$coords[,,i]
	P <- t(N)
	dim(P) <- c(1, 24)	
	align_shapes[i,] <- P	
	}

align_reference <- t(M$consensus)
dim(align_reference) <- c(1, 24)

# Now export to csv files.
write.table(input_shapes, file="input_shapes.csv", sep=",", col.names=F, row.names=F)
write.table(align_shapes, file="align_shapes.csv", sep=",", col.names=F, row.names=F)
write.table(align_reference, file="align_reference.csv", sep=",", col.names=F, row.names=F)