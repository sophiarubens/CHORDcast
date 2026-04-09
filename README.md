**Tips**

,,, coming soon ,,,



**Warnings**
1. If you're annoyed by superfluous verbosity, consider modifying your copy of 
   21cmSense to override the redshift and k-scale extrapolation warnings
   
       * high k extrap warning triggered because the CHORD FoV is so wide
   
       * low z extrap warning triggered because the underlying 21cmFast calls are not well-suited
         to post-EoR surveys
   
       * neither of these pose actual problems because the 21cmFast power spectrum is only used for
         the sample variance term, which I get from my ensembles

       * I only use 21cmSense to get the thermal noise term, which doesn't require a 21cmFast call
