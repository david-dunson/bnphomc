function loglik=logmly(z,Y,M,alpha)
d0=length(unique(Y));                       % let x_{j,1},...,x_{j,p0} be the set of included predictors
[z0,m]=unique(sortrows([Y z]),'rows','legacy');      % z0 are the sorted unique combinations of (y,z_{j,1},...,z_{j,p0}), m contains their positions on {1,...,n}
C=tensor(zeros([d0 M]),[d0 M]);             % d0=levels of the response y, M=number of clustered levels of x_{j,1},...,x_{j,p0}
C(z0)=C(z0)+m-[0;m(1:(end-1))];             % add the differences in positions to cells of clT corresponding to the unique combinations -> gives the number of times (y,z_{j,1},...,z_{j,p0}) appears 
Cdata=tenmat(C,1);                          % matrix representation of the tensor C, with rows of the matrix corresponding to dimension 1 i.e. the levels of y

JJ=size(Cdata);
loglik=sum(sum(gammaln(Cdata.data+alpha)))-sum(gammaln(sum(Cdata.data,1)+d0*alpha))-JJ(2)*(d0*gammaln(alpha)-gammaln(d0*alpha));
