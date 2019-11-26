
d0=4;
qmax=10;     % maximum dependence order
pnew=qmax;
dnew=d0*ones(1,qmax);   % redefine the number of levels of each predictor, now the predictor set includes qmax preceding y values
dmax=max(dnew);
pM=zeros(pnew,dmax);
%varphi_set=[0 0.25 0.5 0.75 1];
varphi_set=[0.5];
n_varphi=length(varphi_set);
for i=1:n_varphi
    varphi=varphi_set(i);
    for j=1:pnew
        pM(j,1:dnew(j))=exp(-(varphi*j*(1:dnew(j))-1)); % prior probability for k_{j}
        pM(j,1:dnew(j))=pM(j,1:dnew(j))./sum(pM(j,1:dnew(j)));
    end

    Nsample=10000;
    Ksample=zeros(Nsample,qmax);
    for j=1:qmax
        Ksample(1:Nsample,j)=randsample(dnew(j),Nsample,'true',pM(j,:));
    end
    Kgrtrthn1=sum((Ksample>1),1)./Nsample;
    NKgrtrthn1=tabulate(sum((Ksample>1),2));

    h(2*i-1)=subplot(n_varphi,2,2*i-1);
    h(2*i)=subplot(n_varphi,2,2*i);

    bar(h(2*i-1),Kgrtrthn1);
    set(gca,'XTick',1:qmax);
    xlim(h(2*i-1),[0 pnew+5]);
    ylim(h(2*i-1),[0 1]);
    str=sprintf('$$\\varphi = %1.2f$$',varphi);
	lh=gtext(str);
	set(lh,'Interpreter','latex');

    bar(h(2*i),NKgrtrthn1(:,3)./100,'BarWidth', 0.7);
    xlim(h(2*i),[0 pnew+2]);
    ylim(h(2*i),[0 0.5]);
    str=sprintf('$$\\varphi = %1.2f$$',varphi);
	lh=gtext(str);
	%lh=title(h(2*i),str);
	set(lh,'Interpreter','latex');
    set(gca,'XTickLabel',0:qmax);
end

