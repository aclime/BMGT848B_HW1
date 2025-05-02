data dsf0; set crsp.dsf; /*ADD DATE VARIABLES TO DSF*/
	y = year(date);
	m = month(date);
	q = qtr(date);
	mdate = intnx('month',mdy(m,1,y),0,'E');
	qdate = intnx('month',mdy(3*q,1,y),0,'E');
	keep permno date y m q mdate qdate ret prc;
run;

proc sql; /*GET RF AND MKTRF FROM FAMA-FRENCH*/
	create table dsf1 as
	select a.*, a.ret-b.rf as exret, b.mktrf, b.rf
	from dsf0 as a, ff.factors_daily as b
	where a.date=b.date;
quit;

proc sort data = dsf1; by permno mdate; run; quit;
%let myt0 = %sysfunc(time(),time8.);
options nonotes nosource errors=1;
proc reg data=dsf1 outest=betas noprint; /*PRE-RANKING BETAS*/
	where prc>5;/*exclude days with low price */
	by permno mdate;
	model exret = mktrf / sse;
run;
%let myt1 = %sysfunc(time(),time8.); %put &myt0; %put &myt1;
data betas; set betas; where _edf_>=15; run; /* DROP IF TOO FEW RETS IN ESTIMATION PERIOD */


proc sql; /* RESTRICT TO COMMON EQUITY */
	create table betas1 as
	select a.*
	from betas as a, crsp.stocknames as b
	where a.permno=b.permno and b.namedt<=a.mdate<=b.nameenddt and b.shrcd in (10,11);
quit;


data msf; set crsp.msf; /*CREATE DATE VARIABLES IN REALIZED RETURN DATA*/
	y = year(date);
	m = month(date);
	q = qtr(date);
	mdate = intnx('month',mdy(m,1,y),0,'E');
	qdate = intnx('month',mdy(3*q,1,y),0,'E');
	mdate1 = intnx('month',mdy(m,1,y),-1,'E');
	qdate1 = intnx('month',mdy(3*q,1,y),-3,'E');
	mktcap = abs(prc)* shrout;
	keep permno date y m q mdate qdate mdate1 qdate1 ret prc mktcap;
run;
data msf; set msf; /*LAGGED MKT CAP*/
	by permno;
	lmktcap = lag(mktcap);
	if first.permno then lmktcap = .;
	run;

proc sql; /* MERGE LAG BETAS WITH NEXT MONTH RETURN */
	create table rets as
	select a.*, b.mktrf as betamkt
	from msf as a, betas1 as b
	where a.permno=b.permno and a.mdate1=b.mdate and a.ret>-2;
quit;
data rets; set rets; where lmktcap>0 and betamkt ne .; run;

/* DOUBLE SORT */
proc sort data = rets; by date; run; quit;
proc rank data=rets groups=5 out=rets;
	by date;
	var lmktcap;
	ranks sizerank;
run;
proc sort data = rets; by date sizerank; run; quit;
proc rank data=rets groups=5 out=rets;
	by date sizerank;
	var betamkt;
	ranks betarank;
run;
data rets; format port $4.; set rets; 
	sizerank = sizerank + 1;
	betarank = betarank + 1;
	port = cats('S',sizerank,'B',betarank);
	run;


proc summary data = rets nway; /* COLLAPSE BY MONTH */
	class date port sizerank betarank;
	var ret;
	weight lmktcap;
	output out = portrets mean=;
	run; quit;
proc sql; /* GET FACTOR RETURNS */
	create table portrets2 as
	select a.*, a.ret-b.rf as exret, b.*
	from portrets as a, ff.factors_monthly as b
	where a.date=intnx('month',b.date,0,'E')
	order by a.date, betarank;
quit;
data portrets2; set portrets2; y = year(date); drop _type_ _freq_; run;

data temp1; set portrets2; where betarank in (1,5); run;
proc sort data = portrets2; by date sizerank betarank; run;

data temp1; set temp1; /*LAGGED MKT CAP*/
	by date sizerank;
	lret = lag(exret);
	if first.date or first.sizerank then lret = .;
	run;
data temp1; set temp1;
	betarank = 0;
	exret = exret - lret;
	port = cats('S',sizerank,'B',betarank);
	where lret ne .;
	drop lret;
	run;
data portrets_final; set portrets2 temp1; run;
proc sort data = portrets_final; by betarank date; run; quit;
