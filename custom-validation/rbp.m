function [R,t,s,rms] = rbp(p1,p2,wf)

%Function returns rigid body transformation parameters between two reference frames:

    % - rotation matrix R

    % - translation vector t

    % - scale factor s

    % - rms values of residuals X, Y and Z coordinate axis

%Input variables:

    %p1, p2 coordinates of initial reference and transformed reference frames respectively

    %p2=s*R*p1+t

    %p1 and p2 are nx3 matrices where n is number of points

    %each row represents X, Y and Z data from one point, i.e.

    %p1=[X1 Y1 Z1; X2 Y2 Z3; ... Xn Yn Zn];

    %Minimum of n=3 noncollinear points are needed

    %wf is nx1 weighting factor vector, usually normalized between 0 and 1

    %input all ones for wf vector if no weighting factors are available

%Algorithm is based on linear least square method and takes advantage of SVD factorization

    %Derivation and theory behind it is represented in the following paper:

    %Challis, J.H. (1995) A procedure for determining rigid body transformation parameters.

    %Journal of Biomechanics 28:5;733-737

%Author:    Tomislav Pribanic, University of Zagreb, Croatia

%e-mail:    tomislav.pribanic@fer.hr

%           Any comments and suggestions are more than welcome.

%Date:      September, 2003

%Version:   1.0

if size(p1,1)<3 | size(p2,1)<3 | size(p1,1)~=size(p2,1) | size(p1,2)~=3 | size(p2,2)~=3 | size(wf,1)~=size(p1,1)

    % size(p1,1)
    % size(p2,1)
    % size(wf,1)
    % size(p1,2)
    % size(p2,2)

    pom{1}='Input coordinates matrices must be equally sized nx3 matrices,';

    pom{2}='where minimum needed number of n noncollinear points is 3!';

    pom{3}='For nx1 weighting factor vector, input all ones if no weighting data is available.';

    % WARNDLG(pom,'Function Abort')

    R=NaN;t=NaN;s=NaN;rms=NaN;

    return

else

    p1old=p1;p2old=p2;  %original data

    wf=wf.^2;  pom=sum(wf);

    p1mean=sum([p1(:,1).*wf p1(:,2).*wf p1(:,3).*wf])/pom; %(Eq. 6, i.e. Eq. 30 in paper)

    p2mean=sum([p2(:,1).*wf p2(:,2).*wf p2(:,3).*wf])/pom; %(Eq. 7, i.e. Eq. 31)

    p1=[p1(:,1)-p1mean(1) p1(:,2)-p1mean(2) p1(:,3)-p1mean(3)];    %(Eq. 10)

    p2=[p2(:,1)-p2mean(1) p2(:,2)-p2mean(2) p2(:,3)-p2mean(3)];    %(Eg. 11)

    n=length(p1);

    C=zeros(3,3);

    for i=1:n %(Eq. 19, i.e. 32)

        C=C+wf(i)*p2(i,:)'*p1(i,:);

    end,C=C/pom;

    [U,S,V]=svd(C);

    R=U*[1 0 0; 0 1 0; 0 0 det(U*V')]*V'; %(Eq. 24)

    %p1mean,p2mean,p1,p2,C,U,V

    sigx=mean(diag(p1(1:end,:)*p1(1:end,:)')); %(Eq. 26c)

    s=trace(R'*C)/sigx; %(Eq. 27)

    t=p2mean'-s*R*p1mean';%(Eq. 28)

    %p2mean'-1*R*p1mean',pause

    %predicted values in second reference frame

    p2pred=s*(R*p1old')';

    p2pred=[p2pred(:,1)+t(1) p2pred(:,2)+t(2) p2pred(:,3)+t(3)];

    %RMS values of residuals

    rms=p2old-p2pred;

    rms=rms.*rms;

    rms=sqrt(sum(rms)/n);
end
