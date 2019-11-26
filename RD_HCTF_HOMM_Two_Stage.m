%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%% -- Conditional Tensor Factorization Based Higher Order Markov Chains -- %%%
%%% --                  for Categorical Sequences                        -- %%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

% -- By Abhra Sarkar, Last modified on Jan, 2015 -- %

% -- This version allows the response to have any number of levels.  %
% -- This version implements a two-stage sampler. The first stage    %
% implements SSVS that implements a hard clustering based model      %
% to determine the values of k_j's. The second stage runs a Gibbs    %
% sampler keeping the k_j's fixed. The core tensors are clustered    %
% using a Dirichlet process prior.                                   %

% -- Download and Install Tensor Toolbox for Matlab -- %
% -- Installation Instructions -- %
% 1. Unpack the files.
% 2. Rename the root directory of the toolbox from tensor_toolbox_2.5 to tensor_toolbox.
% 3. Start MATLAB.
% 4. Within MATLAB, cd to the tensor_toolbox directory and execute the following commands. 
%       addpath(pwd)            %<-- add the tensor toolbox to the MATLAB path
%       cd met; addpath(pwd)    %<-- also add the met directory
%       savepath                %<-- save for future MATLAB sessions


clear all;

seed=9;  
rng(seed);  
RandStream.getGlobalStream;


%%%%%%%%%%%%%%%%%
%%% Load Data %%%
%%%%%%%%%%%%%%%%%

load WoodPewee.mat;
T=1000;                  % training size
N=(T+300);               % total size
y=y(1:N);
dataname='Wood_Pewee';

d0=length(unique(y));   % number of levels of the response
train=1:T;
Y0=y;
Y=y(train);
MMM=tabulate(Y); MMM(:,3)
pause(3);


%%%%%%%%%%%%%%%%%%%%%
%%% Assign Priors %%%
%%%%%%%%%%%%%%%%%%%%%

qmax=10;     % maximum dependence order
p=0;         % number of external predictors
pnew=(qmax+p);
dnew=d0*ones(1,qmax);   % redefine the number of levels of each predictor, now the predictor set includes qmax preceding y values
dmax=max(dnew);
pM=zeros(pnew,dmax);
for j=1:pnew
    pM(j,1:dnew(j))=exp(-(j*(1:dnew(j))/2-1)); % prior probability for k_{j}
    pM(j,1:dnew(j))=pM(j,1:dnew(j))./sum(pM(j,1:dnew(j)));
end
lambdaalpha=1/d0;   % prior for Dirichlet distribution for lambda
pigamma(1:pnew)=1./dnew;     % prior for Dirichlet distribution for pi
ep=max(10,floor(log(T)/log(4))); %expected maximum number of predictors
ep=min(pnew,ep);



%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%% Sampler for First Stage %%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

fprintf('\n--- First Stage --- \n\n');
N1=1000;     % number of iterations for first stage
np=0;       % number of predictors included in the model 
Nnew=N-qmax;
X0new=zeros(Nnew,pnew);                 %                                     ---- Structure of X0new  ---  
Y0new=Y0((qmax+1):N);                   %
for j=1:qmax                            %   Y(qmax+1)->    Y(qmax)   Y(qmax-1)  ...       Y(1)    X(qmax+1,1)     ...   X(qmax+1,p)
    X0new(:,j)=Y0(qmax+1-j:(N-j));      %   Y(qmqx+2)->    Y(qmax+1) Y(qmax)    ...       Y(2)    X(qmax+2,1)     ...   X(qmax+2,p)
end                                     %     ...    ->      ...      ...       ...       ...         ...         ...       ...
for j=(qmax+1):pnew                     %     Y(t)   ->     Y(t-1)   Y(t-2)     ...    Y(t-qmax)    X(t,1)        ...     X(t,p)
    X0new(:,j)=X0((qmax+1):N,j-qmax);   %     ...    ->      ...      ...       ...       ...         ...         ...       ...
end                                     %     Y(N)   ->     Y(N-1)   Y(N-2)    ...     Y(N-qmax)  X(N-qmax,1)    ...   X(N-qmax,p)
Tnew=(T-qmax);
Xnew=X0new(1:Tnew,:);
Ynew=Y0new(1:Tnew);
M=ones(N1+1,pnew);
G=zeros(pnew,dmax);
for j=1:pnew
    G(j,1:dnew(j))=1;
end
z=ones(Tnew,pnew);
log0=zeros(N1,1);
cM=zeros(pnew,dmax);
for j=1:pnew                                % There are (2^r-2)/2 ways to split r objects into two non-empty groups.
    cM(j,1:dnew(j))=(2.^(1:dnew(j))-2)/2;   % (Consider 2 cells and r objects, subtract 2 for the two cases in which 
end                                         % one cell receives all objects, divide by 2 for symmetry.) 
% Let A denote a partition of {1,...,d}.
% The algorithm proceeds as follows. 
% 1. propose new k* and A* according to q(k,A->k*,A*)=1/(no of possible moves with |k-k*|=1)
% 2. accept with MH acceptence probability
%       a={p0(k*)p0(A*|k*)L(Y|k*,A*)q(k*,A*->k,A)}/{p0(k)p0(A|k)L(Y|k,A)q(k,A->k*,A*)}
% We set p(A|k) = 1/(total no of ways of partitioning d objects in |A|=k nonempty sets), i.e uniform over all possibilities. 
% Let gn=gn(A) denote the cluster sizes of the |A| clusters.
% For splits, we have k*=(k+1) and  q(k,A->k*,A*)=1/sum(cM(gn))=1/cM(d), when k=1,       and q(k*,A*->k,A)=1/(k*C2)=1/(2C2),
%                                   q(k,A->k*,A*)=1/sum(cM(gn)),         when 1<k<(d-1), and q(k*,A*->k,A)=1/(k*C2),
%                                   q(k,A->k*,A*)=1/sum(cM(gn))=1/cM(2), when k=(d-1),   and q(k*,A*->k,A)=1/(k*C2)=1/(dC2).       
% For mergers, we have k*=(k-1) and q(k,A->k*,A*)=1/(kC2)=1/(2C2),       when k=2,       and q(k*,A*->k,A)=1/sum(cM(gn*))=1/cM(d),
%                                   q(k,A->k*,A*)=1/(kC2),               when 2<k<d,     and q(k*,A*->k,A)=1/sum(cM(gn*)),
%                                   q(k,A->k*,A*)=1/(kC2)=1/(dC2),       when k=d,       and q(k*,A*->k,A)=1/sum(cM(gn*))=1/cM(2).       
for k=1:N1
    M00=M(k,:);     % initiate M00={k_{1},...,k_{p}}, current values of k_{j}'s for the kth iteration
    for j=1:pnew
        M0=M00(j);  % M0=k_{j}, current value
        if M0==1    % if x_{j} is not included, propose its inclusion OR a switch with an existing important predictor
          % propose the inclusion of x_{j} with randomly generated cluster mappings for different levels of x_{j}
          if np<ep  % propose inclusion of x_{j} only if np<ep so that np never exceeds ep
            new=binornd(1,0.5*ones(1,dnew(j)-1));	% propose new mapping for (d_{j}-1) values at 1
            while sum(new)==0
                new=binornd(1,0.5*ones(1,dnew(j)-1));
            end
            GG=G(j,1:dnew(j))+[0 new]; % keep the first one at 1, propose new cluster mappings for the other (d_{j}-1) levels of x_{j}
            zz=z;                   % zz initiated at z
            zz(:,j)=GG(Xnew(:,j));  % proposed new zz by mapping to new cluster configurations for the observed values of x_{j}
            ind2=find(M00>1);       % current set of relevant predictors
            if isempty(ind2)        % if no predictor is currently important
                ind2=1;
            end
            MM=M00;                 % MM initiated at {k_{1},...,k_{p}}, current values, with k_{j}=1 by the if condition
            MM(j)=2;                % proposed new value of k_{j}=2
            ind1=find(MM>1);        % proposed set of important predictors, now includes x_{j}, since MM(j)=2
            logR=logml(zz(:,ind1),Ynew,MM(ind1),pM(ind1,:),lambdaalpha)-logml(z(:,ind2),Ynew,M00(ind2),pM(ind2,:),lambdaalpha);
            logR=logR+log(cM(j,dnew(j)));
            if log(rand)<logR
                G(j,1:dnew(j))=GG;
                M00=MM;
                z=zz;
                np=np+1;
            end
          end
        end
        if M0>1&&M0<dnew(j) % if 1<k_{j}<d_{j} (recall that M0=M00(j)=k_{j})
            if rand<0.5     % with prob 0.5 merge two mapped values into one
                cnew=randsample(M0,2);
                lnew=max(cnew);
                snew=min(cnew);
                GG=G(j,1:dnew(j));
                GG(GG==lnew)=snew;  % replace all lnews by snews
                GG(GG==M0)=lnew;    % replace the largest cluster mapping by lnew (lnew itself may equal M0, in which case GG remains unchanged by this move)
                zz=z;               % zz initiated at z
                zz(:,j)=GG(Xnew(:,j)); % proposed new z_{tj} as per the proposed new cluster mappings of the levels of x_{j}
                MM=M00;             % MM initizted at current values {k_{1},...,k_{p}}, with k_{j}=d_{j} by the if condition
                MM(j)=M00(j)-1;     % proposed new value of k_{j}, since two mappings are merged
                ind1=find(MM>1);    % proposed set of important predictors, may not include x_{j} if original k_(j) was at 2
                ind2=find(M00>1);   % current set of important predictors
                if isempty(ind1)
                    ind1=1;
                end
                if isempty(ind2)
                    ind2=1;
                end
                logR=logml(zz(:,ind1),Ynew,MM(ind1),pM(ind1,:),lambdaalpha)-logml(z(:,ind2),Ynew,M00(ind2),pM(ind2,:),lambdaalpha);
                if M0>2
                    [z0,mm]=unique(sort(GG),'legacy');
                    gn=mm-[0 mm(1:(end-1))];
                    logR=logR-log(sum(cM(j,gn)))+log(M00(j)*(M00(j)-1)/2);
                else
                    logR=logR-log(cM(j,dnew(j))); 
                end
                if log(rand)<logR
                    G(j,1:dnew(j))=GG;
                    M00=MM;
                    z=zz;
                    if M00(j)==1
                        np=np-1;
                    end
                end
            else        % with prob 0.5 split one mapped value into two
                [z0,mm]=unique(sort(G(j,1:dnew(j))),'legacy');                        % z0 are unique cluster mappings, mm contains their positions
                gn=mm-[0 mm(1:(end-1))];            % frequencies of z0
                pgn=cM(j,gn)/sum(cM(j,gn));         % see the definition of cM
                rr=sum(mnrnd(1,pgn).*(1:M00(j)));   % rr is the state to split, gn(rr) is the frequency of rr
                new=binornd(1,0.5*ones(1,gn(rr)-1));% propose new mapping for (gn(rr)-1) values at rr
                while sum(new)==0
                    new=binornd(1,0.5*ones(1,gn(rr)-1));
                end
                GG=G(j,1:dnew(j));
                GG(GG==rr)=rr+(M0+1-rr)*[0 new]; % keep first value at rr, propose new mapping (M0+1) for the rest of (gn(rr)-1) values at rr
                zz=z;               % zz initiated at z
                zz(:,j)=GG(Xnew(:,j)); % proposed new z_{tj} as per the proposed new cluster mappings of the levels of x_{j}
                MM=M00;             % MM initizted at current values {k_{1},...,k_{p}}
                MM(j)=M00(j)+1;     % proposed new value of k_{j}, since one original mapped value is split into two
                ind1=find(MM>1);    % proposed set of important predictors
                ind2=find(M00>1);   % current set of important predictors
                if isempty(ind2)
                    ind2=1;
                end                
                logR=logml(zz(:,ind1),Ynew,MM(ind1),pM(ind1,:),lambdaalpha)-logml(z(:,ind2),Ynew,M00(ind2),pM(ind2,:),lambdaalpha);
                if M00(j)<dnew(j)-1
                    logR=logR-log(M00(j)*(M00(j)+1)/2)+log(sum(cM(j,gn)));
                else
                    logR=logR-log(dnew(j)*(dnew(j)-1)/2);
                end
                if log(rand)<logR
                    G(j,1:dnew(j))=GG;
                    M00=MM;
                    z=zz;
                end     
            end
        end        
        if M0==dnew(j)              % if M0=k_{j}=d_{j}, propose a merger of two different cluster mappings
            cnew=randsample(dnew(j),2);
            lnew=max(cnew);
            snew=min(cnew);
            GG=G(j,1:dnew(j));
            GG(GG==lnew)=snew;      % replace all lnews by snews
            GG(GG==M0)=lnew;        % replace the largest cluster mapping d_{j} by lnew (lnew itself can be d_{j}, in which case GG remains unchanged by this move)
            zz=z;                   % zz initiated at z
            zz(:,j)=GG(Xnew(:,j));  % proposed new z_{tj} as per the proposed new cluster mappings of the levels of x_{j}
            MM=M00;                 % MM initizted at current values {k_{1},...,k_{p}}, with k_{j}=d_{j} by the if condition
            MM(j)=dnew(j)-1;        % proposed new value of k_{j}, since originally k_{j}=d_{j} and now two mappings are merged
            ind1=find(MM>1);        % proposed set of important predictors, does not include x_{j} when d_(j)=2
            if isempty(ind1)
               ind1=1;
            end
            ind2=find(M00>1);       % current set of important predictors
            if isempty(ind2)
               ind2=1;
            end            
            logR=logml(zz(:,ind1),Ynew,MM(ind1),pM(ind1,:),lambdaalpha)-logml(z(:,ind2),Ynew,M00(ind2),pM(ind2,:),lambdaalpha);
            logR=logR+log(dnew(j)*(dnew(j)-1)/2);
            if log(rand)<logR
                G(j,1:dnew(j))=GG;
                M00=MM;
                z=zz;
            end   
            if M00(j)==1
                np=np-1;
            end
        end 
    end
    M(k+1,:)=M00;
    
    % print informations in each iteration
    ind1=find(M00>1);
    if isempty(ind1)
        ind1=1;
    end   
    log0(k)=logml(z(:,ind1),Ynew,M(k+1,ind1),pM(ind1,:),lambdaalpha);
    [aaY,bY]=find(M(k+1,1:qmax)-1);
    [aaX,bX]=find(M(k+1,(qmax+1):pnew)-1);
    fprintf('k = %i, %i important predictors = {',k,np);
    for i=1:length(bY)
        fprintf(' Y(t-%i)(%i)',bY(i),M(k+1,bY(i)));
    end
    for i=1:length(bX)
        fprintf(' X%i(%i)',bX(i),M(k+1,qmax+bX(i)));
    end
    fprintf(' }. %f \n',log0(k));
end

VarSelect=(M(1:N1,:)>1);
VarSelectProps=sum(VarSelect(floor(N1/2)+1:N1,:),1)./(N1-floor(N1/2));
display(VarSelectProps);    













%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%% Sampler for Second Stage %%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

fprintf('\n --- Second Stage --- \n\n');
pause(1);
simsize=5000;            % number of iterations for second stage
burnin=2000;             %floor(simsize/5);
log01=zeros(simsize,1);
log02=zeros(simsize,1);
nord=zeros(simsize,1);
npred=zeros(simsize,1);
npredact=zeros(simsize,1);
nzstar=zeros(simsize,1);

% Storage for prediction
gap=5;
kgap=1;
Yfut=zeros(N-T,qmax);
for qq=1:qmax
	Yfut(:,qq) = y((T+1-qq):(N-qq));
end
PP=zeros(floor((simsize-burnin)/gap),d0,(N-T));


ind00=find(VarSelectProps>0.5);   % selected predictors for which k_{j}>1
if isempty(ind00)
    ind00=1;
end
M0=M00(ind00);       % k_{j}'s for the selected predictors
p00=length(ind00);   % number of selected predictors
z00=z(:,ind00);
x0=Xnew(:,ind00);    % selected predictor values
d00=dnew(ind00);
dmax=max(d00);       % redefine dmax
pigamma00=pigamma(ind00);
pi=zeros(p00,dmax,dmax);% pi(j,x_{j},k,iteration no), x_{j} runs in {1,...,d_{j}} and k runs in {1,...,k_{j}}, dmax is used for d_{j} and k_{j}, extra values will not be used or updated
    
    
kstar=100;
pigammastar=1;
Vstar=betarnd(ones(1,kstar),pigammastar*ones(1,kstar));
oneminusVstarProd=[1,cumprod(1-Vstar)];
oneminusVstarProd=oneminusVstarProd(1:kstar);
pistar=oneminusVstarProd.*Vstar;
lambdaalphastar=1;
lambdastarmat=gamrnd(repmat(lambdaalphastar,d0,kstar),1);
colsum_lambdastarmat=sum(lambdastarmat,1);
lambdastarmat=bsxfun(@rdivide,lambdastarmat,colsum_lambdastarmat);

for k=1:simsize        
    
    % pi
    cp=zeros(p00,dmax,dmax);   % counts for pi,first=j,second=value of x,third=value of z
    for t=1:Tnew
        for j=1:p00
            cp(j,x0(t,j),z00(t,j))=cp(j,x0(t,j),z00(t,j))+1;
        end
    end
    for j=1:p00
        for s=1:d00(j)
            rr=gamrnd(cp(j,s,1:M0(j))+pigamma00(j),1);
            pi(j,s,1:M0(j))=rr/sum(rr);	% s=x_{j}, only the first k_{j} elements are updated, rest are left at zero
        end
        % switch labels 
        % in the reshaped d_{j}*k_{j} matrix below, the rows correspond to the values of x_{j} and the columns to the values of h_{j}, the sum across each row is one 
        [~, qq2]=sort(sum(reshape(pi(j,:,1:M0(j)),dmax,M0(j)),1),'descend');
        for s=1:d00(j)
            pi(j,s,1:M0(j))=pi(j,s,qq2);    % column labels h_{j}'s are switched
        end
        for t=1:Tnew
            z00(t,j)=find(qq2==z00(t,j));       % z00_{t,j}'s are switched
        end
    end
    
    
    clT=tensor(zeros([d0,M0]),[d0,M0]);         % d0 levels of the response y, M0={k_{1},..,k_{p00}} soft clustered levels of x_{1},...,x_{p00}
    [z0,m]=unique(sortrows([Ynew z00]),'rows','legacy'); % z0 are the sorted unique combinations of (y,z_{1},...,z_{p00}), m contains their positions on {1,...,T}
    clT(z0)=clT(z0)+m-[0;m(1:(end-1))];         % add the differences in positions to cells of clT corresponding to the unique combinations -> gives the number of times (y,z_{1},...,z_{p00}) appears 
    clTdata=tenmat(clT,1);                      % matrix representation of the tensor clT, with rows of the matrix corresponding to dimension 1 i.e. the levels of y
    sz=size(clTdata);
    
    
    % zstar
    zstarmat=zeros(sz(2),kstar);
    for jj=1:sz(2)
        qqpistar=log(pistar)+sum(log(lambdastarmat).*(repmat(clTdata.data(:,jj),1,kstar)),1);
        qqpistar=exp(qqpistar);
        qqpistar=qqpistar/sum(qqpistar);
        if isnan(sum(qqpistar))
            qqpistar=(1/kstar)*ones(1,kstar);
        end
        zstarmat(jj,:)=mnrnd(1,qqpistar,1);
    end
    nstar=sum(zstarmat,1);
    zstar=sum(bsxfun(@times,zstarmat,1:kstar),2)';
    
    
    % pistar
    Vstar=betarnd(1+nstar,pigammastar+sum(nstar)-cumsum(nstar));
    oneminusVstarProd=[1,cumprod(1-Vstar)];
    oneminusVstarProd=oneminusVstarProd(1:kstar);
    pistar=oneminusVstarProd.*Vstar;
    
    
    % lambdastar
    astar=clTdata.data*zstarmat;
    lambdastarmat=gamrnd(astar+lambdaalphastar,1);
    colsum_lambdastarmat=sum(lambdastarmat,1);
    lambdastarmat=bsxfun(@rdivide,lambdastarmat,colsum_lambdastarmat);
    
    
    % lambda
    lambdamat=lambdastarmat(:,zstar);
    lambda=tensor(lambdamat,[d0,M0]);
    
    
    % z
    for j=1:p00
        qq=zeros(Tnew,M0(j)); % for computing p(z=h|-)
        for h=1:M0(j)
            qq(:,h)=pi(j,x0(:,j),h).*(reshape(double(lambda([Ynew,z00(:,1:(j-1)),h*ones(Tnew,1),z00(:,(j+1):p00)])),Tnew,1))';
        end
        qq=bsxfun(@rdivide,qq,(sum(qq,2)));                 % normalize qq
        z00(:,j)=sum(bsxfun(@times,mnrnd(1,qq),1:M0(j)),2); % sample from multinomial(1,qq) and then multiply with {1,...,k_{j}} 
    end                                                     % finally, summing across columns gives z_{tj} since (k_{j}-1) columns are at zero
    
    
    
    Mact=ones(1,pnew);
    for j=1:p00
        Mact(j)=length(unique(z00(:,j)));
    end
    npredact(k)=length(find(Mact>1));
    if npredact(k)>0
        nord(k)=find(Mact>1,1,'last');
    end
    nzstar(k)=length(unique(zstar));

    
    % print informations in each iteration
    log01(k)=logml(z00,Ynew,M00(ind00),pM(ind00,:),lambdaalpha);
    log02(k)=logmly(z00,Ynew,M00(ind00),lambdaalpha);
    
    fprintf('k = %i, %i important predictors = {',k,p00);
    for i=1:p00
        fprintf(' Y(t-%i)(%i)',ind00(i),M0(i));
    end
    fprintf(' }. %f \n',log01(k)); 
    
    
    % prediction
    if k>burnin && mod(k,gap)==0
    U=cell({1:(p00+1)});
    U{1}=diag(ones(1,d0));
    for j=2:(p00+1)
        U{j}=reshape(pi(j-1,1:d00(j-1),1:M0(j-1)),d00(j-1),M0(j-1));
    end
    PPE=tensor(ttensor(tensor(double(lambda),[d0,M0]),U));
    PPEinds=zeros(d0*(N-T),p00+1);
    PPEinds(:,1)=repmat(1:d0,1,(N-T));
    PPEinds(:,2:(p00+1))=kron(Yfut(:,ind00),ones(d0,1));
    PP(kgap,:,:)=reshape(PPE(PPEinds),d0,(N-T));
    kgap=kgap+1;
    end
end


%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%% Evaluate Performance %%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%

tabulate(npredact);

indpred=(T+1):N;
Ypred=zeros(N-T,1);
PPP=reshape(mean(PP,1),d0,(N-T));
for t=1:(N-T)
    [~, Ypred(t)]=max(PPP(:,t));
end
EmpMclError=1-(abs(Y0(indpred)-Ypred)<1);   % misclassification indicators
EmpMclError=mean(EmpMclError);              % misclassification error rate by our model
fprintf('\n');
fprintf('Estimated misclassification error rate = %f.\n',EmpMclError);

filename=sprintf('Plot_%s_Data_Set_%d_%d_%d_%d',dataname,qmax,simsize,burnin,seed);


%%%%%%%%%%%%%%%%%%%%
%%% Draw Figures %%%
%%%%%%%%%%%%%%%%%%%%

if 1==1;
    fig=figure;
    set(fig, 'Visible', 'on');

    h(1)=subplot(4,3,1);
    h(2)=subplot(4,3,4);
    h(3)=subplot(2,3,2);
    h(4)=subplot(2,3,3);
    h(5)=subplot(2,3,4);
    h(6)=subplot(2,3,5);
    h(7)=subplot(2,3,6);

    plot(h(1),log02);
    xlim(h(1),[1 simsize]);
    title(h(1),'(a)','fontsize',15);

    plot(h(2),1:N1,sum(VarSelect,2));
    xlim(h(2),[1 N1]);
    ylim(h(2),[0 pnew+1]);
    title(h(2),'(b)','fontsize',15);
    
    ordTab=tabulate(nord((burnin+1):simsize));
    xxx=1:qmax;
    yyy=ordTab(:,3); yyy(max(ordTab(:,1))+1:qmax)=0;
    yyy=yyy/sum(yyy);
    bar(h(3),xxx,yyy);
    xlim(h(3),[0 pnew+1]);
    ylim(h(3),[0 1]);
    title(h(3),'(c)','fontsize',15);

    predTab=tabulate(npredact((burnin+1):simsize));
    xxx=1:qmax;
    yyy=predTab(:,3); yyy(max(predTab(:,1))+1:qmax)=0;
    yyy=yyy/sum(yyy);
    bar(h(4),xxx,yyy);
    xlim(h(4),[0 pnew+1]);
    title(h(4),'(d)','fontsize',15);

    bar(h(5),1:pnew,VarSelectProps);
    xlim(h(5),[0 pnew+1]);
    title(h(5),'(e)','fontsize',15);

    zstarTab=tabulate(nzstar((burnin+1):simsize));
    xxx=zstarTab(:,1);
    yyy=zstarTab(:,3);
    yyy=yyy/sum(yyy);
    bar(h(6),xxx,yyy);
    xlim(h(6),[find(yyy>0,1)-1,find(yyy>0,1,'last')+1]);
    title(h(6),'(f)','fontsize',15);

    prodMZ=prod(M,2);
    prodMZTab=tabulate(prodMZ);
    xxx=prodMZTab(:,1);
    yyy=prodMZTab(:,3);
    yyy=yyy/sum(yyy);
    yyyinds=find(yyy>0);
    bar(h(7),xxx(yyyinds),yyy(yyyinds),'BarWidth', 1);
    title(h(7),'(g)','fontsize',15);

    %print(fig,filename,'-depsc');
end


