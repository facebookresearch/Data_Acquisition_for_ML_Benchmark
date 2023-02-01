import matplotlib  # noqa
matplotlib.use('Agg')  # noqa

import matplotlib.pyplot as plt
plt.rcParams['axes.facecolor'] = 'white'

import numpy as np
import matplotlib.ticker as ticker
import json
import seaborn as sn
import pandas as pd
from matplotlib.colors import LogNorm
import seaborn as sns
from matplotlib.colors import LinearSegmentedColormap
import umap
#import matplotlib.pyplot as plt


class VisualizeTools(object):
    def __init__(self,figuresize = (10,8),figureformat='jpg',
                 colorset=['r','orange','k','yellow','g','b','k'],
                 markersize=30,
                 fontsize=30,
                 usecommand=True):
        self.figuresize=figuresize
        self.figureformat = figureformat
        self.fontsize = fontsize
        self.linewidth = 5
        self.markersize = markersize
        self.folder = "../figures/" # use "../figures/" if needed
        self.colorset=colorset
        self.markerset = ['o','X','^','v','s','o','*','d','p']
        self.marker = 'o' # from ['X','^','v','s','o','*','d','p'],
        self.linestyle = '-' # from ['-.','--','--','-.','-',':','--','-.'],
        self.linestyleset = ['-','-.','--','--','-.','-',':','--','-.']
        self.usecommand = usecommand
        
    def plotline(self,
                 xvalue,
                 yvalue,
                 xlabel='xlabel',
                 ylabel='ylabel',
                 legend=None,
                 filename='lineplot',
                 fig=None,
                 color=None,
                 ax=None):
        if(ax==None):        
            # setup figures
            fig = plt.figure(figsize=self.figuresize)
            fig, ax = plt.subplots(figsize=self.figuresize)
            plt.rcParams.update({'font.size': self.fontsize})
            plt.rcParams["font.weight"] = "bold"
            plt.rcParams["axes.labelweight"] = "bold"
            plt.rcParams["lines.linewidth"] = self.linewidth
            plt.rcParams["lines.markersize"] = self.markersize
            plt.rcParams["font.sans-serif"] = 'Arial'        

        # plot it        
        if(color==None):
            color = self.colorset[0]
        ax.plot(xvalue, 
                 yvalue,
                 marker=self.marker,
                 label=legend,
                 color=color,
                 linestyle = self.linestyle,
                 zorder=0,
                 )
        plt.xlabel(xlabel)
        plt.ylabel(ylabel) 
        
        plt.grid(True)
        ax.locator_params(axis='x', nbins=6)
        ax.locator_params(axis='y', nbins=6)

        formatter = ticker.FormatStrFormatter('%0.2e')
        
        formatterx = ticker.FormatStrFormatter('%0.2f')
        
        ax.yaxis.set_major_formatter(formatter)
        ax.xaxis.set_major_formatter(formatterx)
     
        filename =filename+'.'+self.figureformat
        
        if(self.figureformat=='jpg'):
            plt.savefig(filename, format=self.figureformat, bbox_inches='tight',dpi=300)       
        else:
            plt.savefig(filename, format=self.figureformat, bbox_inches='tight')       
            
        return fig, ax 
        #plt.fill_between(bud, np.asarray(acc_mean)-np.asarray(acc_std), np.asarray(acc_mean)+np.asarray(acc_std),alpha=0.3,facecolor='lightgray')


    def plotlines(self,
                 xvalue,
                 yvalues,
                 xlabel='xlabel',
                 ylabel='ylabel',
                 legend=None,
                 filename='lineplot',
                 fig=None,
                 ax=None,
                 showlegend=False,
                 log=False,
                 fontsize=60,
                 basey=10,
                 ylim=None):
        #if(-1):
        if(ax==None):        
            # setup figures
            fig = plt.figure(figsize=self.figuresize)
            fig, ax = plt.subplots(figsize=self.figuresize,frameon=True)
            plt.rcParams.update({'font.size': fontsize})
            plt.rcParams["font.weight"] = "bold"
            plt.rcParams["axes.labelweight"] = "bold"
            plt.rcParams["lines.linewidth"] = self.linewidth
            plt.rcParams["lines.markersize"] = self.markersize
            plt.rcParams["font.sans-serif"] = 'Arial'   
            ax.set_facecolor("white")
            #ax.set_edgecolor("black")
            ax.grid("True",color="grey")
            ax.get_yaxis().set_visible(True)
            ax.get_xaxis().set_visible(True)
        # plot it        
        for i in range(len(yvalues)):
            ax.plot(xvalue, 
                 yvalues[i],
                 marker=self.markerset[i],
                 label=legend[i],
                 color=self.colorset[i],
                 linestyle =  self.linestyleset[i],
                 zorder=0,
                 markersize=self.markersize,
                 markevery=1,
                 )
        plt.xlabel(xlabel,fontsize=fontsize)
        plt.ylabel(ylabel,fontsize=fontsize) 
        
        plt.grid(True)
        #ax.locator_params(axis='x', nbins=6)
        #ax.locator_params(axis='y', nbins=6)
        '''
        formatter = ticker.FormatStrFormatter('%d')
        
        formatterx = ticker.FormatStrFormatter('%d')
        
        ax.yaxis.set_major_formatter(formatter)
        ax.xaxis.set_major_formatter(formatterx)
        '''
        ax.tick_params(axis='both', which='major', labelsize=fontsize)
        
        if(ylim!=None):
            plt.ylim(ylim)

        if(log==True):
            ax.set_yscale('log',base=basey)
        if(showlegend==True):
            ax.legend(legend,facecolor="white",prop={'size': fontsize},
                      markerscale=1, numpoints= 2,loc="best")
        
        filename =filename+'.'+self.figureformat
        
        if(self.figureformat=='jpg'):
            plt.savefig(filename, format=self.figureformat, bbox_inches='tight',dpi=300)       
        else:
            plt.savefig(filename, format=self.figureformat, bbox_inches='tight')       
            
        return fig, ax 
        #plt.fill_between(bud, np.asarray(acc_mean)-np.asarray(acc_std), np.asarray(acc_mean)+np.asarray(acc_std),alpha=0.3,facecolor='lightgray')
    
    def Histogram(self,
                 xvalue,

                 xlabel='xlabel',
                 ylabel='ylabel',
                 legend=None,
                 filename='lineplot',
                 fig=None,
                 ax=None,
                 showlegend=False,
                 log=False,
                 fontsize=90,
                 ylim=None,
                 n_bins=20):
        #if(-1):
        if(ax==None):        
            # setup figures
            fig = plt.figure(figsize=self.figuresize)
            fig, ax = plt.subplots(figsize=self.figuresize,frameon=True)
            plt.rcParams.update({'font.size': fontsize})
            plt.rcParams["font.weight"] = "bold"
            plt.rcParams["axes.labelweight"] = "bold"
            plt.rcParams["lines.linewidth"] = self.linewidth
            plt.rcParams["lines.markersize"] = self.markersize
            plt.rcParams["font.sans-serif"] = 'Arial'   
            ax.set_facecolor("white")
            #ax.set_edgecolor("black")
            ax.grid("True",color="grey")
            ax.get_yaxis().set_visible(True)
            ax.get_xaxis().set_visible(True)
        # plot it
        plt.hist(xvalue,bins=n_bins)        
        '''
        for i in range(len(yvalues)):

            ax.plot(xvalue, 
                 yvalues[i],
                 marker=self.markerset[i],
                 label=legend[i],
                 color=self.colorset[i],
                 linestyle =  self.linestyleset[i],
                 zorder=0,
                 markersize=self.markersize,
                 markevery=10,
                 )
        '''
        plt.xlabel(xlabel,fontsize=fontsize)
        plt.ylabel(ylabel,fontsize=fontsize) 
        
        plt.grid(True)
        #ax.locator_params(axis='x', nbins=6)
        #ax.locator_params(axis='y', nbins=6)
        '''
        formatter = ticker.FormatStrFormatter('%d')
        
        formatterx = ticker.FormatStrFormatter('%d')
        
        ax.yaxis.set_major_formatter(formatter)
        ax.xaxis.set_major_formatter(formatterx)
        '''
        ax.tick_params(axis='both', which='major', labelsize=fontsize)
        
        if(ylim!=None):
            plt.ylim(ylim)

        if(log==True):
            ax.set_yscale('log')
        if(showlegend==True):
            ax.legend(legend,facecolor="white",prop={'size': fontsize},
                      markerscale=2, numpoints= 2,loc=0)
        
        filename =filename+'.'+self.figureformat
        
        if(self.figureformat=='jpg'):
            plt.savefig(filename, format=self.figureformat, bbox_inches='tight',dpi=300)       
        else:
            plt.savefig(filename, format=self.figureformat, bbox_inches='tight')       
            
        return fig, ax 
        #plt.fill_between(bud, np.asarray(acc_mean)-np.asarray(acc_std), np.asarray(acc_mean)+np.asarray(acc_std),alpha=0.3,facecolor='lightgray')


    def Histograms(self,
                 xvalues,
                 xlabel='xlabel',
                 ylabel='ylabel',
                 legend=None,
                 filename='lineplot',
                 fig=None,
                 ax=None,
                 showlegend=False,
                 log=False,
                 fontsize=90,
                 color=['red','orange'],
                 ylim=None,
                 n_bins=20):
        #if(-1):
        if(ax==None):        
            # setup figures
            fig = plt.figure(figsize=self.figuresize)
            fig, ax = plt.subplots(figsize=self.figuresize,frameon=True)
            plt.rcParams.update({'font.size': fontsize})
            plt.rcParams["font.weight"] = "bold"
            plt.rcParams["axes.labelweight"] = "bold"
            plt.rcParams["lines.linewidth"] = self.linewidth
            plt.rcParams["lines.markersize"] = self.markersize
            plt.rcParams["font.sans-serif"] = 'Arial'   
            ax.set_facecolor("white")
            #ax.set_edgecolor("black")
            ax.grid("True",color="grey")
            ax.get_yaxis().set_visible(True)
            ax.get_xaxis().set_visible(True)
        # plot it
        plt.hist(xvalues,bins=n_bins, density=True,color=color)        
        '''
        for i in range(len(yvalues)):

            ax.plot(xvalue, 
                 yvalues[i],
                 marker=self.markerset[i],
                 label=legend[i],
                 color=self.colorset[i],
                 linestyle =  self.linestyleset[i],
                 zorder=0,
                 markersize=self.markersize,
                 markevery=10,
                 )
        '''
        plt.xlabel(xlabel,fontsize=fontsize)
        plt.ylabel(ylabel,fontsize=fontsize) 
        
        plt.grid(True)
        #ax.locator_params(axis='x', nbins=6)
        #ax.locator_params(axis='y', nbins=6)
        '''
        formatter = ticker.FormatStrFormatter('%d')
        
        formatterx = ticker.FormatStrFormatter('%d')
        
        ax.yaxis.set_major_formatter(formatter)
        ax.xaxis.set_major_formatter(formatterx)
        '''
        ax.tick_params(axis='both', which='major', labelsize=fontsize)
        
        if(ylim!=None):
            plt.ylim(ylim)

        if(log==True):
            ax.set_yscale('log')
        if(showlegend==True):
            ax.legend(legend,facecolor="white",prop={'size': fontsize},
                      markerscale=2, numpoints= 2,loc=0)
        
        filename =filename+'.'+self.figureformat
        
        if(self.figureformat=='jpg'):
            plt.savefig(filename, format=self.figureformat, bbox_inches='tight',dpi=300)       
        else:
            plt.savefig(filename, format=self.figureformat, bbox_inches='tight')       
            
        return fig, ax 
        #plt.fill_between(bud, np.asarray(acc_mean)-np.asarray(acc_std), np.asarray(acc_mean)+np.asarray(acc_std),alpha=0.3,facecolor='lightgray')
    
        
    def plotscatter(self,
                    xvalue=0.3,
                    yvalue=0.5,
                    filename='lineplot',
                    markersize=10,
                    legend='Learned Thres',
                    color='blue',
                    showlegend=False,
                    fig=None,
                    ax=None):
        if(ax==None):
            # setup figures
            fig = plt.figure(figsize=self.figuresize)
            fig, ax = plt.subplots(figsize=self.figuresize)
            plt.rcParams.update({'font.size': self.fontsize})
            plt.rcParams["font.weight"] = "bold"
            plt.rcParams["axes.labelweight"] = "bold"
            plt.rcParams["lines.linewidth"] = self.linewidth
            plt.rcParams["lines.markersize"] = self.markersize
            plt.rcParams["font.sans-serif"] = 'Arial' 
        
        ax.plot(xvalue,yvalue,'*',markersize=markersize,color=color,
                label=legend)
        if(showlegend):
            handles, labels = ax.get_legend_handles_labels()
            print("labels",labels)
            ax.legend(handles[::-1],labels[::-1], prop={'size': 35},markerscale=3, numpoints= 1,loc=0)


        filename =filename+'.'+self.figureformat
        if(self.figureformat=='jpg'):
            plt.savefig(filename, format=self.figureformat, bbox_inches='tight',dpi=300)       
        else:
            plt.savefig(filename, format=self.figureformat, bbox_inches='tight')       
                    
        return fig, ax


    def plotscatter(self,
                    xvalue=0.3,
                    yvalue=0.5,
                    filename='lineplot',
                    markersize=10,
                    legend='Learned Thres',
                    color='blue',
                    showlegend=False,
                    fig=None,
                    ax=None):
        if(ax==None):
            # setup figures
            fig = plt.figure(figsize=self.figuresize)
            fig, ax = plt.subplots(figsize=self.figuresize)
            plt.rcParams.update({'font.size': self.fontsize})
            plt.rcParams["font.weight"] = "bold"
            plt.rcParams["axes.labelweight"] = "bold"
            plt.rcParams["lines.linewidth"] = self.linewidth
            plt.rcParams["lines.markersize"] = self.markersize
            plt.rcParams["font.sans-serif"] = 'Arial' 
        
        ax.plot(xvalue,yvalue,'*',markersize=markersize,color=color,
                label=legend)
        if(showlegend):
            handles, labels = ax.get_legend_handles_labels()
            print("labels",labels)
            ax.legend(handles[::-1],labels[::-1], prop={'size': 35},markerscale=3, numpoints= 1,loc=0)


        filename =filename+'.'+self.figureformat
        if(self.figureformat=='jpg'):
            plt.savefig(filename, format=self.figureformat, bbox_inches='tight',dpi=300)       
        else:
            plt.savefig(filename, format=self.figureformat, bbox_inches='tight')       
                    
        return fig, ax
    
    def plotscatters_annotation(self,
                    xvalue=[0.3],
                    yvalue=[0.5],
                    filename='lineplot',
                    markersize=10,
                    legend='Learned Thres',
                    color='blue',
                    showlegend=False,
                    fig=None,
                    ax=None,
                    annotation=None):
        if(ax==None):
            # setup figures
            fig = plt.figure(figsize=self.figuresize)
            fig, ax = plt.subplots(figsize=self.figuresize)
            plt.rcParams.update({'font.size': self.fontsize})
            plt.rcParams["font.weight"] = "bold"
            plt.rcParams["axes.labelweight"] = "bold"
            plt.rcParams["lines.linewidth"] = self.linewidth
            plt.rcParams["lines.markersize"] = self.markersize
            plt.rcParams["font.sans-serif"] = 'Arial' 
        
        ax.scatter(xvalue,yvalue,)
        #           '*',markersize=markersize,color=color,
        #        )
        for i in range(len(xvalue)):
            ax.annotate(annotation[i], xy=[xvalue[i],yvalue[i]])
        if(showlegend):
            handles, labels = ax.get_legend_handles_labels()
            print("labels",labels)
            ax.legend(handles[::-1],labels[::-1], prop={'size': 35},markerscale=3, numpoints= 1,loc=0)


        filename =filename+'.'+self.figureformat
        if(self.figureformat=='jpg'):
            plt.savefig(filename, format=self.figureformat, bbox_inches='tight',dpi=300)       
        else:
            plt.savefig(filename, format=self.figureformat, bbox_inches='tight')       
                    
        return fig, ax
    
    def plot_bar(self,barname,barvalue,
                 filename='barplot',
                 markersize=2,
                 yname='Frequency',
                 xname="",
                 color='blue',
                 ylim=None,
                 fig=None,
                 showlegend=False,
                 ax=None,
                 labelpad=None,
                 fontsize=30,
                 threshold=10,
                 add_thresline=False,):
        if(ax==None):
            # setup figures
            fig = plt.figure(figsize=self.figuresize)
            fig, ax = plt.subplots(figsize=self.figuresize)
            ax.set_facecolor("white")
            plt.rcParams.update({'font.size': 1})
            plt.rcParams["font.weight"] = "bold"
            plt.rcParams["axes.labelweight"] = "bold"
            plt.rcParams["lines.linewidth"] = self.linewidth
            plt.rcParams["lines.markersize"] = markersize
            plt.rcParams["font.sans-serif"] = 'Arial' 
            plt.rc('font', size=1)          # controls default text sizes
            print("xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx")
            plt.grid(True,color="grey")
        x = np.arange(len(barname))
        ax.bar(x,barvalue,color=color,
                label=barname)
        ax.set_ylabel(yname,fontsize=fontsize)
        if(xname!=""):
            ax.set_xlabel(xname,fontsize=fontsize)

        #ax.set_title('Scores by group and gender')
        ax.set_xticks(x)
        ax.set_xticklabels(barname,rotation='horizontal',fontsize=fontsize)
        #ax.set_xticklabels(barname,rotation='vertical')
        plt.xlim(x[0]-0.5,x[-1]+0.5)
        
        if(add_thresline==True):
            ax.plot([min(x)-0.5, max(x)+0.5], [threshold, threshold], "k--")

        matplotlib.rc('xtick', labelsize=fontsize) 
        
        ax.tick_params(axis='both', which='major', labelsize=fontsize)

        if(not(labelpad==None)):
            ax.tick_params(axis='x', which='major', pad=labelpad)
      
        #matplotlib.rc('ytick', labelsize=fontsize) 
        #ax.text(0.5,0.5,"hello")

        #ax.legend()
        
        if(showlegend):
            handles, labels = ax.get_legend_handles_labels()
            print("labels",labels)
            ax.legend(handles[::-1],labels[::-1], prop={'size': 10},markerscale=3, numpoints= 1,loc=0)


        #ticks = [tick for tick in plt.gca().get_xticklabels()]
        #print("ticks 0 is",ticks[0].get_window_extent())
        '''
        plt.text(-0.07, -0.145, 'label:', horizontalalignment='center',fontsize=fontsize,
                verticalalignment='center', transform=ax.transAxes)
        plt.text(-0.07, -0.25, 'qs:', horizontalalignment='center',fontsize=fontsize,
                verticalalignment='center', transform=ax.transAxes)
        '''        
        filename =filename+'.'+self.figureformat
        if(not(ylim==None)):
            plt.ylim(ylim)
        if(self.figureformat=='jpg'):
            plt.savefig(filename, format=self.figureformat, bbox_inches='tight',dpi=300)       
        else:
            plt.savefig(filename, format=self.figureformat, bbox_inches='tight')       
                    
        return fig, ax
        
    
    def plot_bar2value(self,barname,barvalue, barvalue2,
                 filename='barplot',
                 markersize=2,
                 yname='Frequency',
                 color='blue',
                 fig=None,
                 showlegend=False,
                 legend=['precision','recall'],
                 yrange = None,
                 ax=None,
                 fontsize=25,
                 showvalues = False,
                 legend_loc="upper left",
                 hatch=None):
        if(ax==None):
            # setup figures
            fig = plt.figure(figsize=self.figuresize)
            fig, ax = plt.subplots(figsize=self.figuresize)
            plt.rcParams.update({'font.size': fontsize})
            plt.rcParams["font.weight"] = "bold"
            plt.rcParams["axes.labelweight"] = "bold"
            plt.rcParams["lines.linewidth"] = self.linewidth
            plt.rcParams["lines.markersize"] = markersize
            plt.rcParams["font.sans-serif"] = 'Arial' 
        width=0.3
        x = np.arange(len(barname))
        ax.bar(x-width/2,barvalue,width,color=color[0],
                label=legend[0])
        ax.bar(x+width/2,barvalue2,width, color=color[1],
               hatch=hatch,
                label=legend[1])
        


        ax.set_ylabel(yname,fontsize=fontsize)
        #ax.set_title('Scores by group and gender')
        ax.set_xticks(x)
        #ax.set_xticklabels(barname,rotation='vertical')
        #ax.set_xticklabels(barname,rotation=45)
        ax.set_xticklabels(barname,rotation='horizontal')
        plt.xlim(x[0]-0.5,x[-1]+0.5)
        if(not(yrange==None)):
            plt.ylim(yrange[0],yrange[1])
            
        matplotlib.rc('xtick', labelsize=fontsize) 
        matplotlib.rc('ytick', labelsize=fontsize) 

        #ax.legend()
        
        if(showvalues==True):
            for i, v in enumerate(barvalue):
                ax.text(i - 0.33,v + 0.1, "{:.1f}".format(v), color=color[0], fontweight='bold',)
    
            for i, v in enumerate(barvalue2):
                ax.text(i + .10,v + 0.2, "{:.1f}".format(v), color=color[1], fontweight='bold',)

        if(showlegend):
            handles, labels = ax.get_legend_handles_labels()
            print("labels",labels)
            ax.legend(handles[::-1],labels[::-1], prop={'size': fontsize},markerscale=3, numpoints= 1,
                      loc=legend_loc,ncol=1, )#bbox_to_anchor=(0, 1.05))


        filename =filename+'.'+self.figureformat
        if(self.figureformat=='jpg'):
            plt.savefig(filename, format=self.figureformat, bbox_inches='tight',dpi=300)       
        else:
            plt.savefig(filename, format=self.figureformat, bbox_inches='tight')       
                    
        return fig, ax
    
    def plotconfusionmaitrix(self,confmatrix,
                             xlabel=None,ylabel=None,
                             filename='confmatrix',
                             keywordsize = 16,
                             font_scale=2,
                             figuresize=(10,10),
                             cmap="coolwarm", # "Blues"
                             vmin=0,
                             vmax=10,
                             fonttype='Arial',
                             title1="",
                             fmt=".1f",
                             xlabel1 = "Predicted label",
                             ylabel1="True label",):
        if(self.usecommand==True):
            return self.plotconfusionmaitrix_common1(confmatrix=confmatrix,
                                                     xlabel=xlabel,
                                                     ylabel=ylabel,
                                                     filename=filename,
                                                     keywordsize = keywordsize,
                                                     font_scale=font_scale,
                                                     figuresize=figuresize,
                                                     cmap=cmap,
                                                     vmin=vmin,
                                                     vmax=vmax,
                                                     fonttype=fonttype,
                                                     title1=title1,
                                                     xlabel1=xlabel1,
                                                     ylabel1=ylabel1,
                                                     fmt=fmt)
        
        sn.set(font=fonttype)
        #boundaries = [0.0, 0.045, 0.05, 0.055, 0.06,0.065,0.07,0.08,0.1,0.15, 1.0]  # custom boundaries
        boundaries = [0.0, 0.06,0.2, 0.25,0.3, 0.4,0.5,0.6,0.7, 0.8, 1.0]  # custom boundaries

        # here I generated twice as many colors, 
        # so that I could prune the boundaries more clearly
        #hex_colors = sns.light_palette('blue', n_colors=len(boundaries) * 2 + 2, as_cmap=False).as_hex()
        #hex_colors = [hex_colors[i] for i in range(0, len(hex_colors), 2)]
        #print("hex",hex_colors)
        # My color
        hex_colors = ['#ffffff','#ebf1f7',
 '#d3e4f3',
 '#bfd8ed',
 '#a1cbe2',
 '#7db8da',
 '#5ca4d0',
 '#3f8fc5',
 '#2676b8',
 '#135fa7',
 '#08488e']
        '''
        ['#e5eff9',
 '#d3e4f3',
 '#bfd8ed',
 '#a1cbe2',
 '#7db8da',
 '#5ca4d0',
 '#3f8fc5',
 '#2676b8',
 '#135fa7',
 '#08488e']
        '''

        boundaries = [0.0, 0.03, 0.06,0.1,0.2,0.29,0.3,0.8,1.0]
        hex_colors = ['#F2F6FA','#ebf1f7','#FFB9C7','#FF1242', '#FF1242','#FF1242','#2676b8','#135fa7','#08488e']
        
        colors=list(zip(boundaries, hex_colors))

        custom_color_map = LinearSegmentedColormap.from_list(
            name='custom_navy',
            colors=colors,
            )

        tol=1e-4
        labels = confmatrix
        confmatrix=confmatrix*(confmatrix>0.35)
        print("confmatrix",confmatrix+tol)
        df_cm = pd.DataFrame(confmatrix+tol,xlabel,ylabel)
        plt.figure(figsize=figuresize)
        sn.set(font_scale=font_scale) # for label size
        g = sn.heatmap(df_cm, 
                       linewidths=0.3,
                       linecolor="grey",
                       cmap=custom_color_map,
                       #annot=True, 
                       annot  = labels,
                       annot_kws={"size": keywordsize},fmt=".1f",
                       #mask=df_cm < 0.02,
                       vmin=vmin+tol,
                       vmax=vmax,
                       cbar=False,
                       #cbar_kws={"ticks":[0.1,0.3,1,3,10]},
                       #norm=LogNorm(),
                       #legend=False,
                       ) # font size
        #g.cax.set_visible(False)
        #sn.heatmap(df, cbar=False) 

        g.set_yticklabels(labels=g.get_yticklabels(), va='center')
        filename =filename+'.'+self.figureformat
        plt.ylabel(ylabel1)
        plt.xlabel(xlabel1)  
        plt.title("Overall accuracy:"+"{:.1f}".format(np.trace(confmatrix)),
                  fontweight="bold",
                  pad=32)
        g.set_xticklabels(g.get_xticklabels(), rotation = 0)
        

        if(self.figureformat=='jpg'):
            plt.savefig(filename, format=self.figureformat, bbox_inches='tight',dpi=300)       
        else:
            plt.savefig(filename, format=self.figureformat, bbox_inches='tight')       
            
        return 0


    def plotconfusionmaitrix_common1(self,confmatrix,
                             xlabel=None,ylabel=None,
                             filename='confmatrix',
                             keywordsize = 16,
                             font_scale=2,
                             figuresize=(10,10),
                             cmap="vlag",
                             vmin=0,
                             vmax=10,
                             fonttype='Arial',
                             title1="",
                             fmt=".1f",
                             xlabel1 = "Predicted label",
                             ylabel1="True label",
                             ):
        print("Use common confusion matrix plot!")
        sn.set(font=fonttype)
        #boundaries = [0.0, 0.045, 0.05, 0.055, 0.06,0.065,0.07,0.08,0.1,0.15, 1.0]  # custom boundaries
        boundaries = [0.0, 0.06,0.2, 0.25,0.3, 0.4,0.5,0.6,0.7, 0.8, 1.0]  # custom boundaries

        # here I generated twice as many colors, 
        # so that I could prune the boundaries more clearly
        #hex_colors = sns.light_palette('blue', n_colors=len(boundaries) * 2 + 2, as_cmap=False).as_hex()
        #hex_colors = [hex_colors[i] for i in range(0, len(hex_colors), 2)]
        #print("hex",hex_colors)
        # My color
        hex_colors = ['#ffffff','#ebf1f7',
 '#d3e4f3',
 '#bfd8ed',
 '#a1cbe2',
 '#7db8da',
 '#5ca4d0',
 '#3f8fc5',
 '#2676b8',
 '#135fa7',
 '#08488e']
        '''
        ['#e5eff9',
 '#d3e4f3',
 '#bfd8ed',
 '#a1cbe2',
 '#7db8da',
 '#5ca4d0',
 '#3f8fc5',
 '#2676b8',
 '#135fa7',
 '#08488e']
        '''

        boundaries = [0.0, 0.03, 0.06,0.1,0.2,0.29,0.3,0.8,1.0]
        hex_colors = ['#F2F6FA','#ebf1f7','#FFB9C7','#FF1242', '#FF1242','#FF1242','#2676b8','#135fa7','#08488e']
        
        colors=list(zip(boundaries, hex_colors))

        custom_color_map = LinearSegmentedColormap.from_list(
            name='custom_navy',
            colors=colors,
            )

        tol=1e-4
        labels = confmatrix
        #confmatrix=confmatrix*(confmatrix>0.35)
        #print("confmatrix",confmatrix+tol)
        df_cm = pd.DataFrame(confmatrix+tol,xlabel,ylabel)
        plt.figure(figsize=figuresize)
        sn.set(font_scale=font_scale) # for label size
        g = sn.heatmap(-df_cm, 
                       linewidths=0.3,
                       linecolor="grey",
                       cmap=cmap,
                       #annot=True, 
                       annot  = labels,
                       annot_kws={"size": keywordsize},fmt=fmt,
                       #mask=df_cm < 0.02,
                       #vmin=vmin+tol,
                       #vmax=vmax,
                       cbar=False,
                       center=0,
                       #cbar_kws={"ticks":[0.1,0.3,1,3,10]},
                       #norm=LogNorm(),
                       #legend=False,
                       ) # font size
        #g.cax.set_visible(False)
        #sn.heatmap(df, cbar=False) 

        g.set_yticklabels(labels=g.get_yticklabels(), va='center')
        filename =filename+'.'+self.figureformat
        plt.ylabel(ylabel1)
        plt.xlabel(xlabel1)  
        print("trece",np.trace(confmatrix),confmatrix)
        plt.title(title1,
                  fontweight="bold", 
                  fontsize=keywordsize*1.1,
                  pad=40)
        g.set_xticklabels(g.get_xticklabels(), rotation = 0)
        

        if(self.figureformat=='jpg'):
            plt.savefig(filename, format=self.figureformat, bbox_inches='tight',dpi=300)       
        else:
            plt.savefig(filename, format=self.figureformat, bbox_inches='tight')       
            
        return 0
    
    def plotconfusionmaitrix_common(self,confmatrix,
                             xlabel=None,ylabel=None,
                             filename='confmatrix',
                             keywordsize = 16,
                             font_scale=2,
                             figuresize=(10,10),
                             cmap='vlag',#sn.diverging_palette(240, 10, n=9),
                             vmin=-5,
                             vmax=10,
                             center=0,
                             fonttype='Arial'):
        
        cmap = LinearSegmentedColormap.from_list('RedWhiteGreen', ['red', 'white', 'green'])


        sn.set(font=fonttype)

        tol=1e-4
        labels = (confmatrix+0.05)*(np.abs(confmatrix)>0.1)
        labels = list()
        for i in range(confmatrix.shape[0]):
            temp = list()
            for j in range(confmatrix.shape[1]):
                a = confmatrix[i,j]
                if(a>0.1):
                    temp.append("+"+"{0:.1f}".format(a))
                if(a<-0.1):
                    temp.append("{0:.1f}".format(a))
                if(a<=0.1 and a>=-0.1):
                    temp.append(str(0.0))                    
            labels.append(temp)
        #labels = (confmatrix+0.05)*(np.abs(confmatrix)>0.1)

        print("labels",labels)

        confmatrix=confmatrix=confmatrix*(np.abs(confmatrix)>0.7)
        
        print("confmatrix",confmatrix+tol)
        df_cm = pd.DataFrame(confmatrix+tol,xlabel,ylabel)
        plt.figure(figsize=figuresize)
        sn.set(font_scale=font_scale) # for label size
        g = sn.heatmap(df_cm, 
                       linewidths=12.0,
                       linecolor="grey",
                       cmap=cmap,
                       center=center,
                       #annot=True, 
                       annot  = labels,
                       annot_kws={"size": keywordsize},fmt="s",#fmt="{0:+.1f}",
                       #mask=df_cm < 0.02,
                       vmin=vmin,
                       vmax=vmax,
                       cbar=False,
                       #cbar_kws={"ticks":[0.1,0.3,1,3,10]},
                       #norm=LogNorm(),
                       #legend=False,
                       ) # font size
        #g.cax.set_visible(False)
        #sn.heatmap(df, cbar=False) 

        g.set_yticklabels(labels=g.get_yticklabels(), va='center')
        filename =filename+'.'+self.figureformat
        plt.ylabel("ML API")
        plt.xlabel("Dataset",)  
        #plt.title("Overall accuracy:"+"{:.1f}".format(np.trace(confmatrix)),
        #          fontweight="bold",
        #          pad=32)
        g.set_xticklabels(g.get_xticklabels(), rotation = 0)
        

        if(self.figureformat=='jpg'):
            plt.savefig(filename, format=self.figureformat, bbox_inches='tight',dpi=40)       
        else:
            plt.savefig(filename, format=self.figureformat, bbox_inches='tight')       
            
        return 0
    
    def reward_vs_confidence(self,
                             BaseID = 100,
                             ModelID=[100,0,1,2],
                             confidencerange = (0.1,0.2,0.3,0.4,0.5,0.6,0.7,.99,1),
                             prob_range=None,
                             datapath='path/to/imagenet/result/val_performance'):
        """
        Run a small experiment on solving a Bernoulli bandit with K slot machines,
        each with a randomly initialized reward probability.
    
        Args:
            K (int): number of slot machiens.
            N (int): number of time steps to try.
        """
        datapath = self.datapath
        print('reward datapath',datapath)
        b0 = BernoulliBanditwithData(ModelID=ModelID,datapath=datapath)
        K = len(ModelID)
        print ("Data generated Bernoulli bandit has reward probabilities:\n", b0.probas)
        print ("The best machine has index: {} and proba: {}".format(
            max(range(K), key=lambda i: b0.probas[i]), max(b0.probas)))
        Params0 = context_params(ModelID=ModelID,datapath=datapath)
        #confidencerange = (0.02,0.03,0.04,0.05,0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,0.95,0.99,0.9999,1)
        #confidencerange = (0.99,0.991,0.992,0.993,0.994,0.995,0.996,0.997,0.9999,1)
        if(not(prob_range==None)):
            confidencerange = self.mlmodels.prob2qvalue(prob_interval=prob_range,conf_id=BaseID)
        BaseAccuracy, Others =self.mlmodels.accuracy_condition_score_List(ScoreRange=confidencerange,BaseID=BaseID,ModelID=ModelID)

        print(BaseAccuracy, Others)
        CDF = Params0.BaseModel.Compute_Prob_vs_Score(ScoreRange=confidencerange)
        print(CDF)
        plot_reward_vs_confidence(confidencerange, BaseAccuracy,Others, ModelID,"model reward compare_ModelID_{}.png".format(ModelID),CDF)

    def reward_vs_prob(self,
                       BaseID = 100,
                       ModelID=[100,0,1,2],
                       confidencerange = (0.1,0.2,0.3,0.4,0.5,0.6,0.7,.99,1),
                       prob_range=None,
                       datapath='path/to/imagenet/result/val_performance',
                       dataname='imagenet_val',
                       context=None):
        """
        compute and plot reward as a function of the probability of not using 
        the basemodel. 
    
        Args:
            See the name.
        """
        datapath = self.datapath
        print('reward datapath',datapath)
        if(not(prob_range==None)):
            confidencerange = self.mlmodels.prob2qvalue(prob_interval=prob_range,conf_id=BaseID,context = context)
        BaseAccuracy, Others =self.mlmodels.accuracy_condition_score_list(ScoreRange=confidencerange,BaseID=BaseID,ModelID=ModelID,context=context)
        print('Base Accuracy', BaseAccuracy, 'Other',Others)
        CDF = self.mlmodels.compute_prob_vs_score(ScoreRange=confidencerange,context = context)
        print('CDF',CDF)
        self._plot_reward_vs_prob(CDF, BaseAccuracy,Others, ModelID,self.folder+"Reward_vs_Prob_BaseID_{}_{}_context_{}.{}".format(BaseID,dataname,context,self.figureformat),CDF)

    def reward_vs_prob_pdf(self,
                       BaseID = 100,
                       ModelID=[100,0,1,2],
                       confidencerange = (0.1,0.2,0.3,0.4,0.5,0.6,0.7,.99,1),
                       prob_range=None,
                       datapath='path/to/imagenet/result/val_performance',
                       dataname='imagenet_val',
                       context=None):
        """
        compute and plot reward as a function of the probability of not using 
        the basemodel. 
    
        Args:
            See the name.
        """
        datapath = self.datapath
        print('reward datapath',datapath)
        if(not(prob_range==None)):
            confidencerange = self.mlmodels.prob2qvalue(prob_interval=prob_range,conf_id=BaseID,context = context)
        BaseAccuracy, Others =self.mlmodels.accuracy_condition_score_list(ScoreRange=confidencerange,BaseID=BaseID,ModelID=ModelID,context=context)
        print('Base Accuracy', BaseAccuracy, 'Other',Others)
        CDF = self.mlmodels.compute_prob_vs_score(ScoreRange=confidencerange,context = context)
        print('CDF',CDF)
        self._plot_reward_vs_prob(CDF, BaseAccuracy,Others, ModelID,self.folder+"Reward_vs_Prob_BaseID_{}_{}_context_{}.{}".format(BaseID,dataname,context,self.figureformat),CDF)

        if(not(prob_range==None)):
            base_pdf,other_pdf = self.mlmodels.accuracy_condition_score_list_cdf2pdf(prob_range,BaseAccuracy,Others,diff = False)
            print('base pdf',base_pdf)
            print('other pdf',other_pdf)
            self._plot_reward_vs_prob(CDF, base_pdf,other_pdf, ModelID,self.folder+"Reward_vs_Probpdf_diff_BaseID_{}_{}_context_{}.{}".format(BaseID,dataname,context,self.figureformat),CDF)
            self._plot_reward_vs_prob(confidencerange, base_pdf,other_pdf, ModelID,self.folder+"Reward_vs_conf_pdf_diff_BaseID_{}_{}_context_{}.{}".format(BaseID,dataname,context,self.figureformat),CDF)


    def qvalue_vs_prob(self,
                       confidence_range = None,
                       BaseID = 100,
                       prob_range = None,
                       dataname = 'imagenet_val',
                       context=None):
        if(not(prob_range==None)):
            confidence_range = self.mlmodels.prob2qvalue(prob_interval=prob_range,conf_id=BaseID,context=context)        
        filename = self.folder+"Conf_vs_prob_BaseID_{}_{}_context_{}.{}".format(BaseID,dataname,context,self.figureformat)
        prob = self.mlmodels.compute_prob_wrt_confidence(confidence_range=confidence_range,BaseID = BaseID,context=context)
        self._plot_q_value_vs_prob(confidence_range,prob,filename)
        return 0
    
    def _plot_reward_vs_prob(self, confidence_range, base_acc, model_acc, model_names, figname, CDF):
        """
        Plot the results by multi-armed bandit solvers.
    
        Args:
            solvers (list<Solver>): All of them should have been fitted.
            solver_names (list<str)
            figname (str)
        """
        fig = plt.figure(figsize=self.figuresize)
        plt.rcParams.update({'font.size': self.fontsize})  
        plt.rcParams["font.weight"] = "bold"
        plt.rcParams["axes.labelweight"] = "bold"   
        plt.rcParams["lines.linewidth"] = self.linewidth
        plt.rcParams["lines.markersize"] = self.markersize
        k=0
        for i in model_acc:
            plt.plot(confidence_range, i, label=model_names[k],marker='x')
            k=k+1
    
        plt.xlabel('Fraction of Low Confidence Data')
        plt.ylabel('Accuracy on Low Confidence Data')
        plt.legend(loc=8, ncol=5)
        plt.savefig(figname, format=self.figureformat, bbox_inches='tight')
    
    def _plot_q_value_vs_prob(self,confidence_range,prob,figname):
        fig = plt.figure(figsize=self.figuresize)
        plt.rcParams.update({'font.size': self.fontsize})
        plt.rcParams["font.weight"] = "bold"
        plt.rcParams["axes.labelweight"] = "bold"
        plt.rcParams["lines.linewidth"] = self.linewidth
        plt.rcParams["lines.markersize"] = self.markersize
        
        plt.plot(prob,confidence_range,marker='x')
        plt.xlabel('Fraction of Low Confidence Data')
        plt.ylabel('Confidence Threshold')
        #plt.legend(loc=9, ncol=5)
        plt.savefig(figname, format=self.figureformat, bbox_inches='tight')
        
    
    def plot_accuracy(self,
                      namestick=['bm', 's0','s1','s2'],
                      model_id=[100,0,1,2],
                      base_id = 100,
                      datapath='path/to/imagenet/result/val_performance',
                      dataname='imagenet_val'):
        datapath = self.datapath
        print('reward datapath',datapath)
        BaseAccuracy, Others =self.mlmodels.accuracy_condition_score_list(ScoreRange=[1],BaseID=base_id,ModelID=model_id)
        print('Base Accuracy', BaseAccuracy, 'Other',len(Others))
        fig = plt.figure(figsize=self.figuresize)
        plt.rcParams.update({'font.size': self.fontsize})
        plt.rcParams["font.weight"] = "bold"
        plt.rcParams["axes.labelweight"] = "bold"
        plt.rcParams["lines.linewidth"] = self.linewidth
        plt.rcParams["lines.markersize"] = self.markersize
        
        flattened = [val for sublist in Others for val in sublist]
        print('flat others',flattened)
        acc = flattened
        #plt.bar(range(len(acc)),acc,color=self.colorset,tick_label=namestick)
        bars = plt.bar(range(len(acc)),acc,color=self.colorset,hatch="/")
        #plt.bar(range(len(acc)),acc,color='r',edgecolor='k',hatch="/")
        #ax = plt.gca()
        #ax.bar(range(1, 5), range(1, 5), color='red', edgecolor='black', hatch="/")
        #patterns = ('-', '+', 'x', '\\', '*', 'o', 'O', '.')
        patterns = ('-', '\\', '/', 'o', 'O', '.','+', 'x','*')
        for bar, pattern in zip(bars, patterns):
            bar.set_hatch(pattern)
        #ax.set_hatch('/')
        plt.xlabel('ML Services')
        plt.ylabel('Accuracy')
        plt.ylim(min(acc)-0.01)
        #set_xticklabels(namestick)
        matplotlib.pyplot.xticks(range(len(acc)), namestick)

        #plt.legend(loc=9, ncol=5)
        figname = self.folder+"accuracy_dataset_{}.{}".format(dataname,self.figureformat)
        plt.savefig(figname, format=self.figureformat, bbox_inches='tight')       
        
    def plot_umaps(self,
                 fit_data=[[1,2,3],[4,5,6]],
                 data=[[1,2,3],[4,5,6]],
                 filename="umap",
                 markersize=2,
                 markershape=["8","s"],
                 yname='Frequency',
                 color=['blue','red'],
                 fig=None,
                 showlegend=False,
                 legend=['male','female'],
                 yrange = None,
                 ax=None,
                 fontsize=30,
                 figureformat="jpg",):
        # generate embeddings
        reducer = umap.UMAP(random_state=42)
        reducer.fit(fit_data[:,0:-1])
        for i in range(len(data)):
            datum1 = data[i]
            embedding = reducer.transform(datum1[:,0:-1])
            plt.scatter(embedding[:, 0], embedding[:, 1], c=datum1[:,-1], cmap='Spectral', s=markersize,marker=markershape[i],label=legend[i])
  
#        plt.legend(loc=8, ncol=5)
        
        lgnd = plt.legend(loc="lower left", scatterpoints=1, fontsize=10)
        for handle in lgnd.legendHandles:
            handle.set_sizes([2.0])
    
        self.figureformat = figureformat
        if(self.figureformat=='jpg'):
            plt.savefig(filename+".jpg", format=self.figureformat, bbox_inches='tight',dpi=300)       
        else:
            plt.savefig(filename, format=self.figureformat, bbox_inches='tight') 
        plt.close("all")
        return 
    
def plot_results(solvers, solver_names, figname):
    """
    Plot the results by multi-armed bandit solvers.

    Args:
        solvers (list<Solver>): All of them should have been fitted.
        solver_names (list<str)
        figname (str)
    """
    assert len(solvers) == len(solver_names)
    assert all(map(lambda s: isinstance(s, Solver), solvers))
    assert all(map(lambda s: len(s.regrets) > 0, solvers))

    b = solvers[0].bandit

    fig = plt.figure(figsize=(14, 4))
    fig.subplots_adjust(bottom=0.3, wspace=0.3)

    ax1 = fig.add_subplot(131)
    ax2 = fig.add_subplot(132)
    ax3 = fig.add_subplot(133)

    # Sub.fig. 1: Regrets in time.
    for i, s in enumerate(solvers):
        ax1.plot(range(len(s.regrets)), s.regrets, label=solver_names[i])

    ax1.set_xlabel('Time step')
    ax1.set_ylabel('Cumulative regret')
    ax1.legend(loc=9, bbox_to_anchor=(1.82, -0.25), ncol=5)
    ax1.grid('k', ls='--', alpha=0.3)

    # Sub.fig. 2: Probabilities estimated by solvers.
    sorted_indices = sorted(range(b.n), key=lambda x: b.probas[x])
    ax2.plot(range(b.n), [b.probas[x] for x in sorted_indices], 'k--', markersize=12)
    for s in solvers:
        ax2.plot(range(b.n), [s.estimated_probas[x] for x in sorted_indices], 'x', markeredgewidth=2)
    ax2.set_xlabel('Actions sorted by ' + r'$\theta$')
    ax2.set_ylabel('Estimated')
    ax2.grid('k', ls='--', alpha=0.3)

    # Sub.fig. 3: Action counts
    for s in solvers:
        ax3.plot(range(b.n), np.array(s.counts) / float(len(solvers[0].regrets)), ls='steps', lw=2)
    ax3.set_xlabel('Actions')
    ax3.set_ylabel('Frac. # trials')
    ax3.grid('k', ls='--', alpha=0.3)

    plt.savefig(figname)

def plot_reward_vs_confidence(confidence_range, base_acc, model_acc, model_names, figname, CDF):
    """
    Plot the results by multi-armed bandit solvers.

    Args:
        solvers (list<Solver>): All of them should have been fitted.
        solver_names (list<str)
        figname (str)
    """
    fig = plt.figure(figsize=(14, 6))
    fig.subplots_adjust(bottom=0.3, wspace=0.3)

    ax1 = fig.add_subplot(121)
    #ax2 = fig.add_subplot(212)
    ax3 = fig.add_subplot(122)
    #ax4 = fig.add_subplot(214)

    # Sub.fig. 1: Regrets in time.
    k=0
    for i in model_acc:
        ax1.plot(confidence_range, i, label=model_names[k],marker='x')
        k=k+1

    ax1.set_xlabel('Probability threshold')
    ax1.set_ylabel('Reward Value')
    ax1.legend(loc=9, bbox_to_anchor=(1.82, -0.25), ncol=5)
    ax1.grid('k', ls='--', alpha=0.3)

    
    # Sub.fig. 2: Regrets in time.
    k=0
    for i in model_acc:
        ax3.plot(confidence_range, CDF, label=model_names[k],marker='x')
        k=k+1

    ax3.set_xlabel('Probability threshold')
    ax3.set_ylabel('CDF')
    ax3.legend(loc=9, bbox_to_anchor=(1.82, -0.25), ncol=5)
    ax3.grid('k', ls='--', alpha=0.3)
    
    
    plt.savefig(figname, dpi=1000)
    
def plot_reward_vs_confidence_old(confidence_range, base_acc, model_acc, model_names, figname, CDF):
    """
    Plot the results by multi-armed bandit solvers.

    Args:
        solvers (list<Solver>): All of them should have been fitted.
        solver_names (list<str)
        figname (str)
    """
    fig = plt.figure(figsize=(14, 4))
    fig.subplots_adjust(bottom=0.3, wspace=0.3)

    ax1 = fig.add_subplot(221)
    ax2 = fig.add_subplot(222)
    ax3 = fig.add_subplot(223)
    ax4 = fig.add_subplot(224)

    # Sub.fig. 1: Regrets in time.
    k=0
    for i in model_acc:
        ax1.plot(confidence_range, i, label=model_names[k],marker='x')
        k=k+1

    ax1.set_xlabel('Probability threshold')
    ax1.set_ylabel('Reward Value')
    ax1.legend(loc=9, bbox_to_anchor=(1.82, -0.25), ncol=5)
    ax1.grid('k', ls='--', alpha=0.3)

    # Sub.fig. 2: Regrets in time.
    k=0
    for i in model_acc:
        ax2.plot(confidence_range, np.array(i)-np.asarray(base_acc), label=model_names[k],marker='x')
        k=k+1

    ax2.set_xlabel('Probability threshold')
    ax2.set_ylabel('Reward Value-Base')
    ax2.legend(loc=9, bbox_to_anchor=(1.82, -0.25), ncol=5)
    ax2.grid('k', ls='--', alpha=0.3)
    
    # Sub.fig. 2: Regrets in time.
    k=0
    for i in model_acc:
        ax3.plot(confidence_range, CDF, label=model_names[k],marker='x')
        k=k+1

    ax3.set_xlabel('Probability threshold')
    ax3.set_ylabel('CDF')
    ax3.legend(loc=9, bbox_to_anchor=(1.82, -0.25), ncol=5)
    ax3.grid('k', ls='--', alpha=0.3)
    
    
    # Sub.fig. 2: Regrets in time.
    k=0
    for i in model_acc:
        ax4.plot(confidence_range, (np.array(i)-np.asarray(base_acc))*np.asarray(CDF), label=model_names[k],marker='x')
        k=k+1

    ax4.set_xlabel('Probability threshold')
    ax4.set_ylabel('Reward*Prob')
    ax4.legend(loc=9, bbox_to_anchor=(1.82, -0.25), ncol=5)
    ax4.grid('k', ls='--', alpha=0.3)

    
    plt.savefig(figname, dpi=1000)
    
    
def reward_vs_confidence(N=1000, 
                         ModelID=[100,0,1,2,3,4],
                         ModelIndex = [0,1,2,3],
                         confidencerange = (0.1,0.2,0.3,0.4,0.5,0.6,0.7,.99,1),
                         datapath='path/to/imagenet/result/val_performance'):
    """
    Run a small experiment on solving a Bernoulli bandit with K slot machines,
    each with a randomly initialized reward probability.

    Args:
        K (int): number of slot machiens.
        N (int): number of time steps to try.
    """
    print('reward datapaht',datapath)
    b0 = BernoulliBanditwithData(ModelID=ModelID,datapath=datapath)
    K = len(ModelID)
    print ("Data generated Bernoulli bandit has reward probabilities:\n", b0.probas)
    print ("The best machine has index: {} and proba: {}".format(
        max(range(K), key=lambda i: b0.probas[i]), max(b0.probas)))
    Params0 = context_params(ModelID=ModelID,datapath=datapath)
    #confidencerange = (0.02,0.03,0.04,0.05,0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,0.95,0.99,0.9999,1)
    #confidencerange = (0.99,0.991,0.992,0.993,0.994,0.995,0.996,0.997,0.9999,1)

    BaseAccuracy, Others = Params0.BaseModel.Compute_Conditional_Accuracy_AmongModel_List(ScoreRange=confidencerange,BaseID=0,ModelID=ModelIndex)
    print(BaseAccuracy, Others)
    CDF = Params0.BaseModel.Compute_Prob_vs_Score(ScoreRange=confidencerange)
    print(CDF)
    
    #CDF1 = Compute_CDF_wrt_Score(ScoreRange=confidencerange)
    #print(CDF1)
    #print(Params0.BaseModel.Compute_Conditional_Accuracy(Score))
    #Params1 = context_params(ModelID=[2])
    #print(Params1.BaseModel.Compute_Conditional_Accuracy(Score))
    
    # Test for different combinaers
    #ParamsTest = BaseModel(ModelID=[0,1,3,4,5,100])
    #output = ParamsTest.Stacking_AllModels()
    # End of Test
    
    # print(ParamsTest.Compute_Conditional_Accuracy_AmongModel(ScoreBound=Score, ModelID = [0,1]))
    plot_reward_vs_confidence(confidencerange, BaseAccuracy,Others, ModelID,"model reward compare_ModelID_{}.png".format(ModelID),CDF)


def test_plotline():
    prange= [0.    , 0.0001, 0.0002, 0.0003, 0.0004, 0.0005, 0.0006, 0.0007,
       0.0008, 0.0009, 0.001 , 0.0011, 0.0012, 0.0013, 0.0014, 0.0015,
       0.0016, 0.0017, 0.0018, 0.0019, 0.002 , 0.0021, 0.0022, 0.0023,
       0.0024, 0.0025, 0.0026, 0.0027, 0.0028, 0.0029, 0.003 , 0.0031,
       0.0032, 0.0033, 0.0034, 0.0035, 0.0036, 0.0037, 0.0038, 0.0039,
       0.004 , 0.0041, 0.0042, 0.0043, 0.0044, 0.0045, 0.0046, 0.0047,
       0.0048, 0.0049, 0.005 , 0.0051, 0.0052, 0.0053, 0.0054, 0.0055,
       0.0056, 0.0057, 0.0058, 0.0059, 0.006 , 0.0061, 0.0062, 0.0063,
       0.0064, 0.0065, 0.0066, 0.0067, 0.0068, 0.0069, 0.007 , 0.0071,
       0.0072, 0.0073, 0.0074, 0.0075, 0.0076, 0.0077, 0.0078, 0.0079,
       0.008 , 0.0081, 0.0082, 0.0083, 0.0084, 0.0085, 0.0086, 0.0087,
       0.0088, 0.0089, 0.009 , 0.0091, 0.0092, 0.0093, 0.0094, 0.0095,
       0.0096, 0.0097, 0.0098, 0.0099, 0.01  , 0.0101, 0.0102, 0.0103,
       0.0104, 0.0105, 0.0106, 0.0107, 0.0108, 0.0109, 0.011 , 0.0111,
       0.0112, 0.0113, 0.0114, 0.0115, 0.0116, 0.0117, 0.0118, 0.0119,
       0.012 , 0.0121, 0.0122, 0.0123, 0.0124, 0.0125, 0.0126, 0.0127,
       0.0128, 0.0129, 0.013 , 0.0131, 0.0132, 0.0133, 0.0134, 0.0135,
       0.0136, 0.0137, 0.0138, 0.0139, 0.014 , 0.0141, 0.0142, 0.0143,
       0.0144, 0.0145, 0.0146, 0.0147, 0.0148, 0.0149, 0.015 , 0.0151,
       0.0152, 0.0153, 0.0154, 0.0155, 0.0156, 0.0157, 0.0158, 0.0159,
       0.016 , 0.0161, 0.0162, 0.0163, 0.0164, 0.0165, 0.0166, 0.0167,
       0.0168, 0.0169, 0.017 , 0.0171, 0.0172, 0.0173, 0.0174, 0.0175,
       0.0176, 0.0177, 0.0178, 0.0179, 0.018 , 0.0181, 0.0182, 0.0183,
       0.0184, 0.0185, 0.0186, 0.0187, 0.0188, 0.0189, 0.019 , 0.0191,
       0.0192, 0.0193, 0.0194, 0.0195, 0.0196, 0.0197, 0.0198, 0.0199,
       0.02  , 0.0201, 0.0202, 0.0203, 0.0204, 0.0205, 0.0206, 0.0207,
       0.0208, 0.0209, 0.021 , 0.0211, 0.0212, 0.0213, 0.0214, 0.0215,
       0.0216, 0.0217, 0.0218, 0.0219, 0.022 , 0.0221, 0.0222, 0.0223,
       0.0224, 0.0225, 0.0226, 0.0227, 0.0228, 0.0229, 0.023 , 0.0231,
       0.0232, 0.0233, 0.0234, 0.0235, 0.0236, 0.0237, 0.0238, 0.0239,
       0.024 , 0.0241, 0.0242, 0.0243, 0.0244, 0.0245, 0.0246, 0.0247,
       0.0248, 0.0249, 0.025 , 0.0251, 0.0252, 0.0253, 0.0254, 0.0255,
       0.0256, 0.0257, 0.0258, 0.0259, 0.026 , 0.0261, 0.0262, 0.0263,
       0.0264, 0.0265, 0.0266, 0.0267, 0.0268, 0.0269, 0.027 , 0.0271,
       0.0272, 0.0273, 0.0274, 0.0275, 0.0276, 0.0277, 0.0278, 0.0279,
       0.028 , 0.0281, 0.0282, 0.0283, 0.0284, 0.0285, 0.0286, 0.0287,
       0.0288, 0.0289, 0.029 , 0.0291, 0.0292, 0.0293, 0.0294, 0.0295,
       0.0296, 0.0297, 0.0298, 0.0299, 0.03  , 0.0301, 0.0302, 0.0303,
       0.0304, 0.0305, 0.0306, 0.0307, 0.0308, 0.0309, 0.031 , 0.0311,
       0.0312, 0.0313, 0.0314, 0.0315, 0.0316, 0.0317, 0.0318, 0.0319,
       0.032 , 0.0321, 0.0322, 0.0323, 0.0324, 0.0325, 0.0326, 0.0327,
       0.0328, 0.0329, 0.033 , 0.0331, 0.0332, 0.0333, 0.0334, 0.0335,
       0.0336, 0.0337, 0.0338, 0.0339, 0.034 , 0.0341, 0.0342, 0.0343,
       0.0344, 0.0345, 0.0346, 0.0347, 0.0348, 0.0349, 0.035 , 0.0351,
       0.0352, 0.0353, 0.0354, 0.0355, 0.0356, 0.0357, 0.0358, 0.0359,
       0.036 , 0.0361, 0.0362, 0.0363, 0.0364, 0.0365, 0.0366, 0.0367,
       0.0368, 0.0369, 0.037 , 0.0371, 0.0372, 0.0373, 0.0374, 0.0375,
       0.0376, 0.0377, 0.0378, 0.0379, 0.038 , 0.0381, 0.0382, 0.0383,
       0.0384, 0.0385, 0.0386, 0.0387, 0.0388, 0.0389, 0.039 , 0.0391,
       0.0392, 0.0393, 0.0394, 0.0395, 0.0396, 0.0397, 0.0398, 0.0399,
       0.04  , 0.0401, 0.0402, 0.0403, 0.0404, 0.0405, 0.0406, 0.0407,
       0.0408, 0.0409, 0.041 , 0.0411, 0.0412, 0.0413, 0.0414, 0.0415,
       0.0416, 0.0417, 0.0418, 0.0419, 0.042 , 0.0421, 0.0422, 0.0423,
       0.0424, 0.0425, 0.0426, 0.0427, 0.0428, 0.0429, 0.043 , 0.0431,
       0.0432, 0.0433, 0.0434, 0.0435, 0.0436, 0.0437, 0.0438, 0.0439,
       0.044 , 0.0441, 0.0442, 0.0443, 0.0444, 0.0445, 0.0446, 0.0447,
       0.0448, 0.0449, 0.045 , 0.0451, 0.0452, 0.0453, 0.0454, 0.0455,
       0.0456, 0.0457, 0.0458, 0.0459, 0.046 , 0.0461, 0.0462, 0.0463,
       0.0464, 0.0465, 0.0466, 0.0467, 0.0468, 0.0469, 0.047 , 0.0471,
       0.0472, 0.0473, 0.0474, 0.0475, 0.0476, 0.0477, 0.0478, 0.0479,
       0.048 , 0.0481, 0.0482, 0.0483, 0.0484, 0.0485, 0.0486, 0.0487,
       0.0488, 0.0489, 0.049 , 0.0491, 0.0492, 0.0493, 0.0494, 0.0495,
       0.0496, 0.0497, 0.0498, 0.0499]

    acc = [0.48301023, 0.48457155, 0.48538639, 0.48615516, 0.48668402,
       0.48743234, 0.48818995, 0.48874007, 0.48916215, 0.48976699,
       0.49029502, 0.49083267, 0.49127285, 0.49186667, 0.49235521,
       0.49291153, 0.49324094, 0.4937676 , 0.494199  , 0.49455204,
       0.49486084, 0.49522269, 0.49560935, 0.49594377, 0.49625499,
       0.49656768, 0.49680171, 0.497076  , 0.49740774, 0.49774282,
       0.49808112, 0.49844063, 0.49888367, 0.49907962, 0.49934593,
       0.4996519 , 0.50010442, 0.50044377, 0.50083441, 0.50119005,
       0.50157951, 0.50191593, 0.50229962, 0.50263862, 0.5029507 ,
       0.50321984, 0.50355179, 0.50382114, 0.50421764, 0.50475099,
       0.50509806, 0.50548435, 0.50571974, 0.50673374, 0.50709485,
       0.50754149, 0.50806022, 0.50838091, 0.50895068, 0.51405688,
       0.51405485, 0.51387681, 0.51375979, 0.51368061, 0.51363966,
       0.51358214, 0.51348813, 0.51320118, 0.5131013 , 0.51299855,
       0.51285864, 0.51261339, 0.51251116, 0.51239189, 0.51230296,
       0.51222542, 0.51213922, 0.51213295, 0.51199515, 0.51189603,
       0.51178252, 0.51170102, 0.51167787, 0.51146198, 0.51132532,
       0.51125551, 0.51083861, 0.51080367, 0.51065056, 0.51054177,
       0.51030458, 0.51009311, 0.50985171, 0.50952144, 0.50941722,
       0.50885601, 0.50872907, 0.5086227 , 0.50837746, 0.50827709,
       0.50811138, 0.50789465, 0.50776248, 0.50757616, 0.50723169,
       0.50710103, 0.50692262, 0.50636731, 0.50551236, 0.5052581 ,
       0.50467229, 0.50438479, 0.50419524, 0.50370731, 0.50344827,
       0.50315895, 0.50299427, 0.50237316, 0.50076062, 0.5000888 ,
       0.49904193, 0.49821103, 0.49782044, 0.49751325, 0.49721562,
       0.49615602, 0.49570051, 0.49546224, 0.49528738, 0.49499799,
       0.49480788, 0.49441211, 0.49400208, 0.4937852 , 0.49357293,
       0.4932995 , 0.49308115, 0.49274721, 0.49232387, 0.4904962 ,
       0.48979254, 0.48883763, 0.48723269, 0.48694961, 0.48664716,
       0.48383779, 0.4823387 , 0.48139062, 0.48097353, 0.48045641,
       0.47893606, 0.47857627, 0.4783139 , 0.47800683, 0.47767765,
       0.47749323, 0.47730572, 0.47711734, 0.47697098, 0.47674   ,
       0.47655673, 0.47617975, 0.47604766, 0.47593491, 0.4756809 ,
       0.47553382, 0.47541119, 0.47521898, 0.47502894, 0.47485214,
       0.47467929, 0.47456147, 0.47439464, 0.4742843 , 0.47419802,
       0.47407429, 0.47394389, 0.47376382, 0.47366419, 0.47347226,
       0.4733965 , 0.47327365, 0.47314522, 0.47295317, 0.47277204,
       0.47265397, 0.47254881, 0.47238721, 0.47231294, 0.47213071,
       0.47205963, 0.471965  , 0.47181219, 0.47166599, 0.4715344 ,
       0.47142736, 0.47133969, 0.47124544, 0.47120949, 0.47109536,
       0.47099978, 0.47084196, 0.47067891, 0.47054779, 0.47039573,
       0.47028735, 0.47016451, 0.47002245, 0.46977837, 0.46963242,
       0.46943925, 0.4692959 , 0.46914154, 0.46891011, 0.4687833 ,
       0.4685379 , 0.46843594, 0.46825524, 0.46778678, 0.46757136,
       0.46737609, 0.46692911, 0.46674504, 0.46645814, 0.46626084,
       0.46601046, 0.46587982, 0.46568659, 0.46549668, 0.46531255,
       0.46491423, 0.4644362 , 0.46398542, 0.4631161 , 0.46295977,
       0.46250332, 0.46236719, 0.46221666, 0.462093  , 0.46187842,
       0.46174634, 0.46159738, 0.46147783, 0.46137749, 0.46129638,
       0.4611781 , 0.46107324, 0.46094401, 0.46083739, 0.46074101,
       0.46072508, 0.46064278, 0.46052262, 0.46042853, 0.46034242,
       0.46028446, 0.46017712, 0.46011206, 0.46002659, 0.45995817,
       0.45986543, 0.45975698, 0.45968683, 0.45957428, 0.45942207,
       0.45930791, 0.45921235, 0.45910849, 0.45898494, 0.45888329,
       0.45879647, 0.45870982, 0.45870496, 0.45862491, 0.45850992,
       0.45846477, 0.4583252 , 0.45870034, 0.45860152, 0.4584608 ,
       0.45840916, 0.45837632, 0.45829484, 0.45822002, 0.45816921,
       0.45808426, 0.45801872, 0.4579592 , 0.45785556, 0.45777885,
       0.4577343 , 0.45766358, 0.45753936, 0.45752268, 0.45744507,
       0.45736837, 0.45728324, 0.45717934, 0.45703663, 0.45697995,
       0.45691548, 0.45679727, 0.45673414, 0.45666303, 0.45661996,
       0.4565089 , 0.45641751, 0.45633791, 0.45626128, 0.45619948,
       0.4561366 , 0.45613471, 0.45607387, 0.45597782, 0.45588608,
       0.45581065, 0.45568215, 0.4555245 , 0.45539021, 0.45530577,
       0.45521037, 0.4550916 , 0.45500052, 0.45498943, 0.45484803,
       0.45476247, 0.45469974, 0.45461052, 0.45449327, 0.45441162,
       0.4543233 , 0.45421517, 0.45414812, 0.45402163, 0.45396933,
       0.45382181, 0.45372327, 0.45364773, 0.4535485 , 0.45345609,
       0.45338647, 0.45332349, 0.45321917, 0.45318078, 0.45311913,
       0.45302852, 0.45289496, 0.45282775, 0.45291292, 0.45281203,
       0.45271895, 0.45259684, 0.45251492, 0.45226131, 0.45199698,
       0.45190208, 0.45177381, 0.45167107, 0.45156732, 0.45120557,
       0.4510243 , 0.45040894, 0.45016372, 0.41494005, 0.41482359,
       0.4147391 , 0.41467827, 0.41456255, 0.41442845, 0.41435356,
       0.41427217, 0.4141186 , 0.41393056, 0.41373277, 0.41356792,
       0.41346815, 0.41313181, 0.41306098, 0.41297357, 0.41284036,
       0.41271761, 0.41264731, 0.41260986, 0.41259229, 0.41252037,
       0.41246792, 0.41244859, 0.41239455, 0.41236259, 0.41230149,
       0.41226418, 0.41217959, 0.41212254, 0.41211362, 0.41207712,
       0.41202834, 0.4119794 , 0.41189217, 0.41186648, 0.41183323,
       0.41177104, 0.4117605 , 0.41172562, 0.41171102, 0.4116806 ,
       0.41165032, 0.41161321, 0.41153588, 0.4114937 , 0.41145179,
       0.41141475, 0.41141205, 0.4113842 , 0.41137095, 0.41133905,
       0.41131634, 0.41129309, 0.41124033, 0.41121707, 0.41119274,
       0.41117111, 0.41115895, 0.41114137, 0.4111238 , 0.4111119 ,
       0.41109377, 0.41106132, 0.41101536, 0.41100238, 0.41097399,
       0.41095669, 0.4109064 , 0.41086747, 0.4108653 , 0.41084692,
       0.41080381, 0.41078624, 0.4107565 , 0.41074001, 0.4107346 ,
       0.41071432, 0.41067972, 0.41063105, 0.41062294, 0.41059725,
       0.41055453, 0.41050722, 0.41047964, 0.41046612, 0.41040232,
       0.41038609, 0.41036176, 0.41036446, 0.41036176, 0.41034473,
       0.41029336, 0.41027285, 0.4102012 , 0.41018011, 0.41015145,
       0.41014199, 0.41010603, 0.4100817 , 0.41002357, 0.40999707,
       0.40999301, 0.40998193, 0.40995883, 0.40995234, 0.40991009,
       0.40989792, 0.4098425 , 0.40983087, 0.40981059, 0.40980789,
       0.40978626, 0.40978414, 0.40976115, 0.40971627, 0.40970445,
       0.40969383, 0.40966004, 0.40961732, 0.40958487, 0.4095454 ,
       0.40952512, 0.40952269, 0.40948619, 0.40948078, 0.40944969,
       0.40944428, 0.40941995, 0.40940778, 0.40941589, 0.40941589,
       0.40937777, 0.40934938, 0.40932234, 0.40931288, 0.4092899 ]
    cost = [5.99998378, 5.99995133, 5.99998378, 5.99998378, 5.99998378,
       5.99998378, 5.99996756, 5.99996756, 5.99998378, 5.99993511,
       6.        , 5.99995133, 5.99998378, 5.99995133, 5.99996756,
       5.99996756, 5.99993511, 5.99998378, 5.99996756, 5.99993511,
       5.99995133, 5.99996756, 5.99993511, 6.        , 5.99998378,
       5.99995133, 6.        , 6.        , 5.99996756, 5.99996756,
       6.        , 5.99995133, 5.99995133, 5.99998378, 5.99991889,
       5.99998378, 5.99995133, 5.99996756, 5.99995133, 5.99991889,
       5.99995133, 6.        , 5.99995133, 6.        , 5.99996756,
       5.99998378, 5.99998378, 5.99993511, 5.99996756, 6.        ,
       5.99993511, 5.99996756, 6.        , 5.99998378, 5.99996756,
       5.99993511, 5.99995133, 5.99996756, 5.99995133, 5.99482512,
       5.96959964, 5.93986438, 5.90917202, 5.875365  , 5.84858218,
       5.81982026, 5.76072286, 5.73048472, 5.70715723, 5.68048796,
       5.65459737, 5.62457011, 5.59642463, 5.57118292, 5.54910454,
       5.52188372, 5.49866978, 5.47579651, 5.45198235, 5.42672442,
       5.39849783, 5.37371034, 5.34908507, 5.32488158, 5.29626565,
       5.27175394, 5.22610473, 5.19873791, 5.17562131, 5.1535267 ,
       5.12797677, 5.10221595, 5.07600091, 5.04115567, 5.01468107,
       4.94257349, 4.91944066, 4.89812472, 4.87351567, 4.84459153,
       4.82238336, 4.79530855, 4.77207839, 4.73781714, 4.70809811,
       4.68477062, 4.65748491, 4.60662838, 4.53348258, 4.5040231 ,
       4.43806372, 4.41154046, 4.3859743 , 4.34853352, 4.32293492,
       4.29568166, 4.26372396, 4.20829278, 4.08779443, 4.03799234,
       3.97743495, 3.91929466, 3.89564272, 3.86397703, 3.83818376,
       3.76805529, 3.72065408, 3.69499059, 3.67588086, 3.65414314,
       3.63321653, 3.60901304, 3.58041334, 3.55827007, 3.53622413,
       3.51049575, 3.48917981, 3.46007722, 3.42815197, 3.31135228,
       3.26297774, 3.19297904, 3.06952826, 3.04595743, 3.01523263,
       2.82674713, 2.74002336, 2.67805464, 2.6509636 , 2.60555772,
       2.50275777, 2.48069561, 2.4648952 , 2.44812147, 2.43077996,
       2.41721822, 2.40394848, 2.39111673, 2.37920966, 2.36603725,
       2.35312439, 2.33891376, 2.32945623, 2.32024203, 2.30869184,
       2.29655765, 2.28491013, 2.27579326, 2.26568685, 2.25663487,
       2.24586334, 2.23767114, 2.22874895, 2.22080008, 2.21041788,
       2.20259879, 2.19516904, 2.18631173, 2.17881708, 2.1691811 ,
       2.16051846, 2.15156382, 2.14377717, 2.13540653, 2.12697099,
       2.12012524, 2.1119817 , 2.10146973, 2.09397508, 2.08388489,
       2.07486536, 2.06678671, 2.05770229, 2.05040231, 2.04141522,
       2.0323308 , 2.02389527, 2.01721173, 2.00828953, 2.00157355,
       1.99427357, 1.98590293, 1.97717539, 1.96831808, 1.96066122,
       1.95154435, 1.94359548, 1.93636039, 1.92682175, 1.915012  ,
       1.90612225, 1.89590228, 1.8862663 , 1.87789566, 1.86851924,
       1.85755305, 1.84786841, 1.83615599, 1.80666407, 1.79229122,
       1.78366102, 1.76315619, 1.75384466, 1.73528648, 1.7239634 ,
       1.71507365, 1.70748167, 1.69943547, 1.69190838, 1.6825644 ,
       1.6716631 , 1.65424048, 1.63678541, 1.61277659, 1.60628772,
       1.5939913 , 1.58695088, 1.57871001, 1.57163714, 1.56274739,
       1.55638829, 1.5488612 , 1.54246966, 1.53666212, 1.53244436,
       1.52702615, 1.52173772, 1.51602751, 1.51041464, 1.50561287,
       1.50136266, 1.49763156, 1.4931218 , 1.48887159, 1.48410226,
       1.48027383, 1.47514762, 1.47080008, 1.46742586, 1.46398676,
       1.459769  , 1.45636234, 1.45321524, 1.44925702, 1.4444877 ,
       1.44023749, 1.43653884, 1.43186685, 1.42761664, 1.42278243,
       1.41905133, 1.41447667, 1.41042113, 1.4068198 , 1.40192071,
       1.39640517, 1.39127896, 1.37959899, 1.37586789, 1.37070923,
       1.36668613, 1.36263059, 1.35948349, 1.35607683, 1.35208617,
       1.34708974, 1.34361819, 1.33988709, 1.33547466, 1.33115956,
       1.32713646, 1.32080981, 1.31717604, 1.31415872, 1.31110895,
       1.30760496, 1.30423074, 1.29998053, 1.29560055, 1.29199922,
       1.28856012, 1.2840828 , 1.28074103, 1.27694504, 1.27065083,
       1.26717929, 1.2636753 , 1.26036597, 1.25686198, 1.25364999,
       1.25004867, 1.24761534, 1.2440789 , 1.24031536, 1.23525404,
       1.23204205, 1.22814872, 1.22266563, 1.2176692 , 1.21319188,
       1.20839011, 1.2038479 , 1.20112257, 1.19677503, 1.19310882,
       1.18992927, 1.18730128, 1.18363507, 1.17850886, 1.17562131,
       1.17302576, 1.16926222, 1.16702355, 1.16189735, 1.15858802,
       1.15313737, 1.14856271, 1.14583739, 1.14340406, 1.13844008,
       1.13526053, 1.13045876, 1.12695477, 1.12267212, 1.11946013,
       1.11400947, 1.10949971, 1.10661216, 1.09973396, 1.09558108,
       1.08763221, 1.08305756, 1.07887223, 1.07186425, 1.06485627,
       1.06021673, 1.05340341, 1.0491532 , 1.04516255, 1.02744793,
       1.02125105, 1.00470443, 0.99208358, 0.23337227, 0.22814872,
       0.22406074, 0.22204919, 0.21922653, 0.2165012 , 0.21445721,
       0.21192655, 0.20916878, 0.20537279, 0.2015768 , 0.19693725,
       0.19512037, 0.18824216, 0.18661995, 0.18389462, 0.18120174,
       0.17740575, 0.17432354, 0.17273376, 0.17205243, 0.17017066,
       0.16861333, 0.167932  , 0.16637467, 0.16484978, 0.16410356,
       0.1630329 , 0.16167024, 0.16014535, 0.1587178 , 0.15787425,
       0.15654403, 0.1552787 , 0.15287781, 0.15209915, 0.15080138,
       0.14963338, 0.14866005, 0.14797872, 0.14778405, 0.14729738,
       0.14642139, 0.14535072, 0.14382584, 0.14324184, 0.14256051,
       0.1414574 , 0.1411654 , 0.1402894 , 0.13951074, 0.13834274,
       0.13727208, 0.13649341, 0.13542275, 0.13493608, 0.13376809,
       0.13250276, 0.13201609, 0.13143209, 0.13065343, 0.13006943,
       0.12932321, 0.12864188, 0.12718188, 0.12659788, 0.12552722,
       0.12429434, 0.12302901, 0.12150412, 0.12101746, 0.12004412,
       0.11926546, 0.11897346, 0.11800013, 0.11735124, 0.11715658,
       0.11647524, 0.11559925, 0.11452858, 0.11407436, 0.11349036,
       0.11261437, 0.11164104, 0.11099215, 0.11044059, 0.10943482,
       0.10875349, 0.10787749, 0.10748816, 0.10651483, 0.10602816,
       0.10505483, 0.10447083, 0.10333528, 0.10297839, 0.10229706,
       0.1018104 , 0.10125884, 0.10057751, 0.09953929, 0.09866329,
       0.09827396, 0.09788463, 0.0974953 , 0.09710596, 0.0960353 ,
       0.09528908, 0.09444553, 0.09366686, 0.09308286, 0.09279086,
       0.09220687, 0.09181753, 0.09152553, 0.0905522 , 0.08987087,
       0.08948154, 0.08883265, 0.08805399, 0.08746999, 0.08669132,
       0.08630199, 0.08581533, 0.084842  , 0.084258  , 0.083674  ,
       0.08321978, 0.08273311, 0.08224645, 0.08205178, 0.08205178,
       0.08127312, 0.08078645, 0.08039712, 0.08010512, 0.07955357]
    a = VisualizeTools()
    max_x = 200
    prange=prange[0:max_x]
    acc =acc[0:max_x]
    cost = cost[0:max_x]
    fig, ax = a.plotline(prange,acc,xlabel='Weight Value', ylabel='Accuracy',
               filename='coco_p_value_acc')
    fig, ax = a.plotscatter(xvalue=[0.0060416667],
                            yvalue=[0.5140010157426727],
                            fig=fig,ax=ax,
                            markersize=30,
                            legend='Learned Thres',                            
                            filename='coco_p_value_acc')
    fig, ax = a.plotline(prange,cost,xlabel='Weight Value', ylabel='Cost',
               filename='coco_p_value_cost')    
    fig, ax = a.plotscatter(xvalue=[0.0060416667],
                            yvalue=[5.9999899999999995],
                            fig=fig,ax=ax,
                            markersize=30,
                            legend='Learned Thres',
                            filename='coco_p_value_cost')    
    
    
def getlabeldist(datapath='..\APIperformance\mlserviceperformance_coco\Model0_TrueLabel.txt'):
    mydict = dict()
    labels = json.load(open(datapath))
    for imgname in labels:
        labelexist = dict()
        for temp in labels[imgname]:
            #print(temp)
            label = temp['transcription']
            if label in mydict:
                if(label not in labelexist):
                    mydict[label]+=1
                    labelexist[label] = 1
            else:
                mydict[label] = 1
    len_img = len(labels)
    return mydict, len_img

def test_label_dist():
    showlegend = True
    a = VisualizeTools(figuresize=(22,8),figureformat='jpg')
    name = ['Microsoft','Google']
    value1 = [5175/6358,4302/6358]
    value2 = [5368/6358,4304/6358]
    legend = ['2020 March', '2021 Feb']
    
    a.plot_bar2value(barname = name,barvalue = value1, 
                     barvalue2 = value2,
                     color=['r','b'],
                     filename='FERPLUS',yname='',
                     legend=legend,
                     showlegend=showlegend,
                     yrange=[min(value1)-0.05,max(value2)+0.05])
       
    
    showlegend = True
    a = VisualizeTools(figuresize=(22,8),figureformat='jpg')
    name = ['Microsoft','Google']
    value1 = [10996/15339,10069/15339]
    value2 = [11000/15339,10073/15339]
    legend = ['2020 March', '2021 Feb']
    
    a.plot_bar2value(barname = name,barvalue = value1, 
                     barvalue2 = value2,
                     color=['r','b'],
                     filename='RAFDB',yname='',
                     legend=legend,
                     showlegend=showlegend,
                     yrange=[min(value1)-0.05,max(value2)+0.05])
        
    
    a.plot_bar(barname = name,barvalue = value1)
    
def getlabelprecisionandrecall(targetlabel='person',
                               truelabelpath='..\APIperformance\mlserviceperformance_coco\Model2_TrueLabel.txt',
                               predlabelpath='..\APIperformance\mlserviceperformance_coco\Model6_PredictedLabel.txt',):
    truelabel = json.load(open(truelabelpath))
    predlabel = json.load(open(predlabelpath))

    count = 0
    for imgname in truelabel:
        truehas = False
        for temp in truelabel[imgname]:
            #print(temp)
            label = temp['transcription']
            if label == targetlabel:
                truehas = True
        predhas = False
        for temp in predlabel[imgname]:
            #print(temp)
            label = temp['transcription']
            if label == targetlabel:
                predhas = True
        if(truehas and predhas):
            count+=1
                
    totaltrue = getlabeldist(truelabelpath)
    totalpred = getlabeldist(predlabelpath)
    if(targetlabel in totalpred[0]):
        pred1 = totalpred[0][targetlabel]
    else:
        pred1 = 0
    print('total true, total pred, all correct',totaltrue[0][targetlabel],pred1,count)

    if(pred1==0):
        return 0, count/totaltrue[0][targetlabel]
    return count/totalpred[0][targetlabel], count/totaltrue[0][targetlabel]    
    
def test_precisionrecall(predlabelpath='cocoresult\majvote_coco.txt',
                         labelid=100,
                         showlegend=False):
    labeldist, labelen = getlabeldist()
    labellist = list()
    precisionlist = list()
    recalllist = list()
    for label in sorted(labeldist):
        print(label)
        pre, recall = getlabelprecisionandrecall(targetlabel=label,
                                                 predlabelpath=predlabelpath,)
        precisionlist.append(pre)
        recalllist.append(recall)
        labellist.append(label)
    print('pre and recall',precisionlist, recalllist)
    np.savetxt('precision'+str(labelid)+'.txt', precisionlist)
    np.savetxt('recall'+str(labelid)+'.txt', precisionlist)
    np.savetxt('label'+str(labelid)+'.txt',labellist,fmt='%s')
    a = VisualizeTools(figuresize=(23,8),figureformat='eps')
    a.plot_bar(barname = labellist,barvalue = precisionlist,filename='precisionmajvote',yname='')
    a.plot_bar(barname = labellist,barvalue = recalllist,filename='recallmajvote',yname='')

    a.plot_bar2value(barname = labellist,barvalue = precisionlist, 
                     barvalue2 = recalllist,
                     color=['r','b'],
                     filename='preandrecall'+str(labelid),yname='',
                     showlegend=showlegend)


    return 0

if __name__ == '__main__':
    '''
    test_precisionrecall(predlabelpath='cocoresult\\FrugalMCTcoco.txt',
                         labelid=99999)   
    test_precisionrecall(predlabelpath='cocoresult\\majvote_coco.txt',
                         labelid=888)
    test_precisionrecall(predlabelpath='cocoresult\\100000_coco_thres.txt',
                         labelid=100000,showlegend=True)
    test_precisionrecall(predlabelpath='cocoresult\\0_coco_thres.txt',
                         labelid=0)
    test_precisionrecall(predlabelpath='cocoresult\\6_coco_thres.txt',
                         labelid=6)    
    test_precisionrecall(predlabelpath='cocoresult\\2_coco_thres.txt',
                         labelid=2)
    ''' 
    
    #getlabelprecisionandrecall()
    
    
    test_label_dist()
    #test_plotline()
    matplotlib.pyplot.close('all')