"""
This file defines the Optimal Pricing Functions
"""

import math
import numpy as np

def zScore2NDs(y1,y2,s1,s2,n1,n2):
    """Two distributions: y=mean, s=std dev, n=number of data points"""
    """
    Written by Glenn Melzer on:   27 May 2015
      Last update:                27 May 2015

    This function generates a modified Z Score to determine if two normal
    curves have statistically different gaussian distributions.  It
    considers not only the mean and population of the two distributions,
    but it also considers the standard deviation of each distribution.

    INPUTS:
      y1 = Mean of distribution 1
      s1 = Standard deviation of distribution 1
      n1 = Population size of distribution 1
      y2 = Mean of distribution 2
      s2 = Standard deviation of distribution 2
      n2 = Population size of distribution 2

    OUTPUT:
      z = the z score

    FORMULAS USED:
      z = (sqrt((y1-y2-s1+s2)^2 + 2*(y1-y2)^2 + (y1-y2+s1-s2)^2)/2 / sqrt((s1^2/n1)+(s2^2/Ns))
     """
    assert (n1 > 0),'Distribution population n1 must be > zero'
    assert (n2 > 0),'Distribution population n2 must be > zero'
    y1 = 1.0 * y1
    s1 = 1.0 * s1
    top = math.sqrt((y1 - y2 - s1 + s2)**2 + 2*(y1 - y2)**2 + (y1 - y2 +s1 - s2)**2) / 2
    bot = math.sqrt((s1 * s1 / n1) + (s2 * s2 / n2))
    zScore = top / bot
    return zScore
    


def LMHErrNum(L, M, H):
    """L=low price, M=median price, H=high price"""
    """
    Written by Glenn Melzer on:   31 Aug 2006
      Last update:                09 May 2015

    This function determines if the placement of Low, Median, and High values
    for Customized Optimal Pricing are in the proper order and the spacing
    between them is not too skewed.  If not, provide an error message number.

    INPUTS:
      L = Low Price (the price that yields a 95% odds of winning)
      M = Median Price (the price that yields a 50% odds of winning)
      H = High Price (the price that yields a 5% odds of winning)
    
    OUTPUT:  (Error Message Numbers)
      0 = No Error
      1 = High value must be greater than the Low value
      2 = High value must be greater than the Median value
      3 = Low value must be less than the Median value
      4 = Median value is too close to the High value
      5 = Median value is too close to the Low value
    
    INTERNAL CONSTANTS:
      delta = the maximum skew (M getting too close to L or H)
    """
    Delta = 0.05
    LMHErrNum = 0
    if L == M and M == H and H == 0:
        LMHErrNum = 0
    elif H <= L:
        LMHErrNum = 1
    elif H <= M:
        LMHErrNum = 2
    elif L >= M:
        LMHErrNum = 3
    elif abs(1.0 *(H - M) / (H - L)) <= Delta:
        LMHErrNum = 4
    elif abs(1.0 * (M - L) / (H - L)) <= Delta:
        LMHErrNum = 5
    return LMHErrNum;

def ErrMsg(N):
    """ N=error message number from LMHErrNum function """
    """
    Written by Glenn Melzer on:   06 Nov 2006
      Last update:                06 May 2015

    This function prints the error message associated with the
    error number returned by the LMHErrNum function.

    INPUT:  The error message number from LMHErrNum function
      
    OUTPUT:  (N -> Error Message Text)
      0 -> (no error message given)
      1 -> High value must be greater than the Low value
      2 -> High value must be greater than the Median value
      3 -> Low value must be less than the Median value
      4 -> Median value is too close to the High value
      5 -> Median value is too close to the Low value
    """
    ErrMsg = ''
    if N == 1:
        ErrMsg = 'High value must be greater than the Low value'
    elif N == 2:
        ErrMsg = 'High value must be greater than the Median value'
    elif N == 3:
        ErrMsg = 'Low value must be less than the Median value'
    elif N == 4:
        ErrMsg = 'Median value is too close to the High value'
    elif N == 5:
        ErrMsg = 'Median value is too close to the Low value'
    print ErrMsg
    return;
 

def CumProb(S):
    """ S=number of standard deviations """
    """
    Written by Glenn Melzer on:   25 Aug 2006
      Last update:                06 May 2015

    This function calculates the cumulative probablility under a normal
    bell curve as a function of the distance from zero measured in
    standard deviations.

    INPUT:
      S (The number of standard deviations)
      
    OUTPUT:
      Q (The cumulative probability)
          [NOTE: in limit, large S->0%, large negative S->100%]
    
    FORMULAS USED:
      Z(S) = (e^(-(S^2)/2))/sqrt(2Pi)
      t(S) = 1/(1 + (p * abs(S)))
      q(S) = Z(S) * (t(b1 +t(b2+t(b3+t(b4+t*b5))))
      Q(S) = If S<0 then Q(S)= 1 - Q(S)
    This function takes S (the number of standard deviations from zero
    and calculates the area under a normal bell curve to the right of S.
    Example: If S=0, then CumProb=50%; if S=1.644853, then CumProb=5%,
    if S=-1.644853, then CumProb=95%.
    
    CONSTANTS REQUIRED:
      sqrt(2Pi) =  2.506628275
      P         =  0.231641900
      b1        =  0.319381530
      b2        = -0.356563782
      b3        =  1.781477937
      b4        = -1.821255978
      b5        =  1.330274429
    """
    zs = (2.7182818285 ** (-(S * S) / 2)) / 2.506628275
    ts = 1 / (1 + 0.2316419 * abs(S))
    qps = zs * (ts * (0.31938153 + ts * (-0.356563782 + ts * (1.781477937 + ts * (-1.821255978 + ts * 1.330274429)))))
    if S < 0:
        CumProb = 1 - qps
    else:
        CumProb = qps
    return CumProb;


def LinTrans(P, L, M, H):
    """P=price input, L=low price, M=median price, H=high price"""
    """
    Written by Glenn Melzer on:   25 Aug 2006
      Last update:                07 May 2015
    
    This function transforms a point P into the number of standard
    deviations it is from the mean.  It is based on a normal probability
    curve that uses Low price (95% probability of winning),
    Median (50% prob), and High (5% prob).
    NOTE:  This transformation is linear and assumes that (H - M) = (M - L)
    
    INPUT:
      P (the price to be probability tested)
      L (Low price - the price that yeilds a 95% chance of winning)
      M (Median price - the price that yeilds a 50% chance of winning)
      H (High price - the price that yeilds a 5% chance of winning)
      
    OUTPUT:
      S (the number of Standard Deviations from the mean (or median))
    
    FORMULAS USED:
      m1 = C1 / (H - M)
      b1 = m1 * M
      S = m1 * P + b1
   
    CONSTANTS REQUIRED:
    c1 = 1.644853475  'The std dev for a 5% cumulative probability.
   
    NOTES:
    Positive S values are to the right of the mean; negative to the left
    If P = L, then LinTrans = -1.644853475; if P = M, then LinTrans = 0;
    if P = H then LinTrans = 1.644853475.  All other values of P are
    translated linearly.
    """
    c1 = 1.644853475
    m1 = c1 / (H - M)
    b1 = -m1 * M
    LinTrans = (m1 * P) + b1
    return LinTrans


def HyperTrans(P, L, M, H):
    """P=price input, L=low price, M=median price, H=high price"""
    """
    Written by Glenn Melzer on:   25 Aug 2006
      Last update:                09 Apr 2014
    
    This function transforms a point P into the number of standard
    deviations it is from the mean.  It is based on a normal probability
    curve that uses Low price (95% probability of winning),
    Median price (50% prob), and High price(5% prob).
    
    Notes:  This transformation is non-linear and assumes that (H - M) <> (M - L)
    It assumes that the data best fit a hyperbolically skewed normal bell
    curve.  This transformation removes the skew from the data and outputs
    the standard deviation from the mean so normal bell curve cumulative
    probability analysis is possible.
    If P = L, then HyperTrans = -1.644853475; if P = M, then HyperTrans = 0; if P = H
    then HyperTrans = 1.644853475.  All other translations of P fall on a hyperbolic
    curve defined by the points (L,-1.644853475), (M,0), and (H,1.644853475).
    
    The function may not stable when:
         (P < L)   or   (P > H)
    In these cases the S value may flip to the other half of the hyperbola.
    This usually doesn't happen unless the curve is highly skewed - when M
    is relatively close to L or M.  Logic has been added after the hyperbolic
    skew to ensure that this potential anomoly is prevented.
    
    INPUT:
      P (the price to be probability tested)
      L (Low price - the price that yeilds a 95% chance of winning)
      M (Median price - the price that yeilds a 50% chance of winning)
      H (High price - the price that yeilds a 5% chance of winning)
   
    OUTPUT:
      S (the number of Standard Deviations from the mean (or median))
    
    FORMULAS USED:
      B = (2HL - M(H + L))/(2M - H - L)
      A = c1((M + B)(H + B))/(M - H)
      C = -A / (M + B)
      S = (A / (P + B)) + C
    
    CONSTANTS REQUIRED:
    c1 = 1.644853475  'The std dev for a 5% cumulative probability.
    """
    c1 = 1.644853475
    B = (2.0 * H * L - M * (H + L)) / (2 * M - H - L)
    A = c1 * ((M + B) * (H + B)) / (M - H)
    C = -A / (M + B)
    temp = (A / (P + B)) + C
    HyperTrans = temp
    
    #Logic added to prevent unstable results:
    if (P < L) and (temp > c1):
        HyperTrans = -c1 * ((M - P) / (M - L))
    elif (P > H) and (temp < c1):
        HyperTrans = c1 * ((P - M) / (H - M))
    return HyperTrans
    

def BestTrans(P, L, M, H):
    """P=price input, L=low price, M=median price, H=high price"""
    """
    Written by Glenn Melzer on:   25 Aug 2006
      Last update:                07 May 2015
    
    This function performs either a linear or hyperbolic transformation depending on
    whether the H, M, and L data is linear or not.
    
    INPUT:
      P (the price to be probability tested)
      L (Low price - the price that yeilds a 95% chance of winning)
      M (Median price - the price that yeilds a 50% chance of winning)
      H (High price - the price that yeilds a 5% chance of winning)
    
    OUTPUT:
      S (the number of Standard Deviations from the mean (or median))
    """
    L1 = (1.0 * (H - M) / (M - L)) - 1
    if abs(L1) <= 0.0005:
        BestTrans = LinTrans(P, L, M, H)
    else:
        BestTrans = HyperTrans(P, L, M, H)
    return BestTrans


def ProbOfWin(P, L, M, H):
    """P=price input, L=low price, M=median price, H=high price"""
    """
    Written by Glenn Melzer on:   25 Aug 2006
      Last update:                07 May 2015
    
    This function finds the probability of winning a deal using price P given a
    marketplace defined with a market probability curve of Low, Median, and High.
    
    INPUT:
      P (the price to be probability tested)
      L (Low price - the price that yeilds a 95% chance of winning)
      M (Median price - the price that yeilds a 50% chance of winning)
      H (High price - the price that yeilds a 5% chance of winning)
    
    OUTPUT:
      Q (the win probability)
    """
    S = BestTrans(P, L, M, H)
    ProbOfWin = CumProb(S)
    return ProbOfWin


def ExpFC(P, L, M, H, Cf, Cv, FCw, FCl):
    """P=price input, L=low price, M=median price, H=high price, Cf=fixed cost, Cv=variable cost(%ofnet), FCw=fin contrib if win, FCl=fin contrib if loss"""
    """
    Written by Glenn Melzer on:   25 Aug 2006
      Last update:                10 Apr 2014
    
    This function calculates the expected financial contribution of a particular
    price given the fixed cost, the variable cost,the probability of winning
    (from Low, Median, and High price data), and the follow-on incremental expected
    financial contribution of winning and losing.
    
    INPUT:
      P   (the price to be evaluated)
      L   (Low price - the price that yeilds a 95% chance of winning)
      M   (Median price - the price that yeilds a 50% chance of winning)
      H   (High price - the price that yeilds a 5% chance of winning)
      Cf  (the fixed cost)
      Cv  (the variable cost)
      FCw (the incremental financial contribution if the deal WINS)
      FCl (the incremental financial contribution if the deal LOSES)
       
    OUTPUT:
      ExpFC (the expected financial contribution)
    """ 
    Q = ProbOfWin(P, L, M, H)
    ExpFC = Q * (P * (1 - Cv) - Cf + FCw) + (1 - Q) * FCl
    return ExpFC


def OptPrice(L, M, H, Cf, Cv, FCw, FCl):
    """L=low price, M=median price, H=high price, Cf=fixed cost, Cv=variable cost(%ofnet), FCw=fin contrib if win, FCl=fin contrib if loss"""
    """
    Written by Glenn Melzer on:   25 Aug 2006
      Last update:                11 Jan 2016

    This function calculates the optimal price by maximizing the expected financial
    contribution of the deal.  The probability of winning is a fuction of the Low,
    Median, and High prices.  The financial contribution is based on price minus
    fixed and variable costs.  There is also an adjustment for follow-on
    incremental expected financial contribution of winning and loosing.

    INPUT:
      L   (Low price - the price that yeilds a 95% chance of winning)
      M   (Median price - the price that yeilds a 50% chance of winning)
      H   (High price - the price that yeilds a 5% chance of winning)
      Cf  (the fixed cost)
      Cv  (the variable cost)
      FCw (the incremental financial contribution if the deal WINS)
      FCl (the incremental financial contribution if the deal LOSES)
    
    OUTPUT:
      OptPrice (the financial contribution maximizing price given the above inputs)
    """
    iterations = 8
    slices = 10
    Start = L - (M - L) / 5.0
    Finish = H + (H - M) / 5.0
    for i in range(iterations):
        GPMax = -99999999.0
        POpt = -99999999.0
        d = (Finish - Start) / slices
        #print Start, Finish, d
        for P in np.arange(Start, Finish, d):
            EGP = ExpFC(P, L, M, H, Cf, Cv, FCw, FCl)
            if EGP > GPMax:
                GPMax = EGP
                POpt = P
        Start = POpt - d
        Finish = POpt + d
    if Cf > (POpt * (1 - Cv)):
        POpt = Cf / (1 - Cv)
    OptPrice = POpt
    return OptPrice



def PriceConv(Lt, Mt, Ht, LISTt, Pt, Li, Mi, Hi, LISTi):
    """Lt=Tot low price, Mt=Tot med price, Ht=Tot high price, LISTt=Tot list price, Pt=Tot price, Li=Low price, Mi=Med price, Hi=High price, LISTi=List price"""
    """
    Written by Glenn Melzer on:   30 Aug 2006
      Last update:                11 Jan 2016
    
    This function transforms price P along the 0,L,M,H,List price curve to an equivelent
    price Pi along the 0,Li,Mi,Hi,Listi price curve.  (a linear transformation)
    This function is used to convert the effect of changing a bottom line price into
    the individual component price changes.

    INPUT:
      Lt    (Low price of Total Deal - the price that yeilds a 95% chance of winning)
      Mt    (Median price of Total Deal - the price that yeilds a 50% chance of winning)
      Ht    (High price of Total Deal- the price that yeilds a 5% chance of winning)
      LISTt (the List Price of Total Deal)
      Pt    (a new price of total deal)
      Li    (Low price of a deal component - the price that yeilds a 95% chance of winning)
      Mi    (Median price of deal component - the price that yeilds a 50% chance of winning)
      Hi    (High price of deal component - the price that yeilds a 5% chance of winning)
      LISTi (List price of a deal component)

    OUTPUT:
      Pi (a new price of a deal component)
    """
    if (Pt < Lt):
        A=0; B=Lt; a=0; b=Li
    elif (Lt <= Pt and Pt < Mt):
        A=Lt; B=Mt; a=Li; b=Mi
    elif (Mt <= Pt and Pt < Ht):
        A=Mt; B=Ht; a=Mi; b=Hi
    elif (Ht <= Pt):
        A=Ht; B=LISTt; a=Hi; b=LISTi
    Pi = 1.0 * a + ((Pt - A) * (b - a)) / (B - A)    
    return Pi

def PriceAdj(low, med, high):
    """low=Low Price(% of Ref), med=Median Price(% of Ref), high=High Price(% of Ref)"""
    """
     Written by Glenn Melzer on:  22 Feb 2016
      Last update:                10 Jun 2016
    
    This function ensures that the Low, Median, and High prices (as percents
    of reference) are in the correct order and Median price is not too close to
    the Low or High price.
    
    INPUT:
      low   (Low price(% of Ref) - the price that yeilds a 95% chance of winning)
      med   (Median price(% of Ref) - the price that yeilds a 50% chance of winning)
      high  (High price(% of Ref) - the price that yeilds a 5% chance of winning)
    
    OUTPUT:
      low   (Low price(% of Ref) - the price that yeilds a 95% chance of winning)
      med   (Median price(% of Ref) - the price that yeilds a 50% chance of winning)
      high  (High price(% of Ref) - the price that yeilds a 5% chance of winning)
    """    
    min = .005 #this is the minimum value of low
    bound = .051 #the Median may not be closer than this to either Low or High
    max = 1.2 #this is the maximum value of High
    
    # this makes any needed adjustments to the low, med, and high price points to eliminate anomolies
    low = np.maximum(np.minimum(low,med), min) #ensures low is not too close to zero
    high = np.minimum(np.maximum(med,high), max) #ensures high isn't too large
    if (high == low): #ensures high <> low
        high = low * (1 + bound)
    med = np.maximum(low, np.minimum(high, med)) #ensures med is between high and low
    if ((high - med) / (high - low)) < bound: #Med too close to High
        med = high - bound * (high - low)
    elif ((med - low) / (high - low)) < bound: #Med too close to Low
        med = low + bound * (high - low)
    return low, med, high
 
   
def OptPriceConfIntervl(OptPrc, L, M, H, Cf, Cv=0, FCw=0, FCl=0, Pct=.95 ):
    """OptPrc=Optimal Price, L=low price, M=median price, H=high price, Cf=fixed cost, Cv=variable cost(%ofnet), FCw=fin contrib if win, FCl=fin contrib if loss, Pct=% of expected GP at Optimal Price"""
    """
    Written by:   Glenn Melzer on:   18 Mar 2016
    Last update:  Glenn Melzer on:   30 Jun 2016              
    
    This function calculates the price range confidence interval around the
    optimal price.  The approach is based on the idea that the optimal price
    maximizes the expected profit contribution given the price uncertainty 
    defined by the L, M, and H price points.  This function assumes that the 
    user may be willing to accept a different price than the optimal price as long
    as the different price doesn't imply a significantly lower expected profit
    contribution.  The Pct value indicates how much lower the expected profit
    contribution can be.  Pct has a default value of .95, meaning that the 
    confidence interval price points (one below the optimal price and one above
    the optimal price) will have an expected profit contribution of 95% of the
    optimal price's profit contribution.  The purpose of these confidence 
    interval price points is to communicate to the user how much flexibility
    the user has in straying off the optimal price before it significantly 
    affects the deal.  Some deals with have broad confidence intervals and
    some will be narrow.
    
    INPUT:
      OptPrc (Optimal Price - around which is calculated the interval)
      L   (Low price - the price that yeilds a 95% chance of winning)
      M   (Median price - the price that yeilds a 50% chance of winning)
      H   (High price - the price that yeilds a 5% chance of winning)
      Cf  (the fixed cost)
      Cv  (the variable cost) [default=0]
      FCw (the incremental financial contribution if the deal WINS) [default=0]
      FCl (the incremental financial contribution if the deal LOSES) [default=0]
      Pct (the percent of the Optimal Price's expected GP that defines the interval) [default=.95]
    
    OUTPUT:
      ConfPriceLow  (the confidence interval price below the optimal price)
      ConfPriceHigh (the confidence interval price above the optimal price)
    """
    #this define the iterations and number of slices to find the confidence price point
    iterations = 8
    slices = 10
    
    #this section is for finding the lower confidence interval price point
    #  this defines the range in which to find the confidence interval price point
    Finish = OptPrc
    Start = min(OptPrc,L) - (M - L) / 5.0
    OptPrcEGP = ExpFC(OptPrc, L, M, H, Cf, Cv, FCw, FCl)
    #  this searches through the slices within the interations to find the interval price point
    for i in range(iterations):
        GPMax = -99999999.0
        PConf = -99999999.0
        d = (Finish - Start) / slices
        #print Start, Finish, d
        for P in np.arange(Start, Finish, d):
            ConfEGP = -abs(ExpFC(P, L, M, H, Cf, Cv, FCw, FCl) - (Pct * OptPrcEGP))
            if ConfEGP > GPMax:
                GPMax = ConfEGP
                PConf = P
        Start = PConf - d
        Finish = min(PConf + d, OptPrc)
    ConfPriceLow = PConf
    
    #this section is for finding the higher confidence interval price point
    #  this defines the range in which to find the confidence interval price point
    Start = OptPrc
    Finish = max(OptPrc,H) + (H - M) / 5.0
    OptPrcEGP = ExpFC(OptPrc, L, M, H, Cf, Cv, FCw, FCl)
    #  this searches through the slices within the interations to find the interval price point
    for i in range(iterations):
        GPMax = -99999999.0
        PConf = -99999999.0
        d = (Finish - Start) / slices
        #print Start, Finish, d
        for P in np.arange(Start, Finish, d):
            ConfEGP = -abs(ExpFC(P, L, M, H, Cf, Cv, FCw, FCl) - (Pct * OptPrcEGP))
            if ConfEGP > GPMax:
                GPMax = ConfEGP
                PConf = P
        Start = max(PConf - d, OptPrc)
        Finish = PConf + d
    ConfPriceHigh = PConf
       
    #this adjusts the bounds to be > 1% different than the optimal price    
    #print ConfPriceLow, OptPrc, ConfPriceHigh
    if ConfPriceLow > (OptPrc * .99): #this forces the lower bound to be at least 1% below the optimal price
        ConfPriceLow = OptPrc * .99
        print 'Lower bound adjusted to be just below the optimal price for this compnent'
    if ConfPriceHigh < (OptPrc * 1.01): #this forces the higher bound to be at least 1% above the optimal price
        ConfPriceHigh = OptPrc * 1.01
        print 'Higher bound adjusted to be just above the optimal price for this compnent'
        
    
    #this section print out the interval prices and statistics
    #print 'Low Interval Price  =', ConfPriceLow
    #print 'EGP                 =', ExpFC(ConfPriceLow, L, M, H, Cf, Cv, FCw, FCl)
    #print '% EGP               =', ExpFC(ConfPriceLow, L, M, H, Cf, Cv, FCw, FCl) / OptPrcEGP
    #print; print 'Optimal price       =', OptPrc
    #print; print 'High Interval Price =', ConfPriceHigh
    #print 'EGP                 =', ExpFC(ConfPriceHigh, L, M, H, Cf, Cv, FCw, FCl)
    #print '% EGP               =', ExpFC(ConfPriceHigh, L, M, H, Cf, Cv, FCw, FCl) / OptPrcEGP
    
    return ConfPriceLow, ConfPriceHigh