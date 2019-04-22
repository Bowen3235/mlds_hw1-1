
var NowposX = 0,NowposY=0,Scroll=true;

function showlist(id){
    var stat=document.getElementsByClassName(id)[0];
    if( stat.style.display == "block" ){
        stat.style.display = "none";
    }
    else{
    	stat.style.display = "block";
    }
}
function return_scroll(){
	var x = document.getElementsByClassName("imgBackGround")[0];

}

function Moveimg( Flag ){
	var x = document.getElementsByClassName("imgBackGround")[0];
	var xc = document.getElementsByClassName("imgcontainer")[0];
	
	if( Flag == "Right" ){
		if( xc.scrollLeft <= Math.round(window.innerWidth*150-50) ) {
			xc.scrollLeft+=25;
		}
		x.style.transform = "translate(" + NowposX +"vw,"+NowposY+"vh)";
		//document.getElementById( "demo" ).innerHTML = "translate(" + NowposX +"%,"+NowposY+"%)";
	}
	else if( Flag == "Left" ){
		if( xc.scrollLeft >= 25 ) {
			xc.scrollLeft-=25;
		}
		x.style.transform = "translate(" + NowposX +"vw,"+NowposY+"vh)";
		//document.getElementById( "demo" ).innerHTML = "translate(" + NowposX +"%,"+NowposY+"%)";
	}
	else if( Flag == "Up" ){
		if( xc.scrollTop >= 25 ) {
			xc.scrollTop-=25;
		}
		x.style.transform = "translate(" + NowposX +"vw,"+NowposY+"vh)";
		//document.getElementById( "demo" ).innerHTML = "translate(" + NowposX +"%,"+NowposY+"%)";
	}
	else if( Flag == "Down" ){
		if( xc.scrollTop <= Math.round(window.innerWidth*250/1436*1072-window.innerHeight-50) ) {
			xc.scrollTop+=25;
		}
		x.style.transform = "translate(" + NowposX +"vw,"+NowposY+"vh)";
		//document.getElementById( "demo" ).innerHTML = "translate(" + NowposX +"%,"+NowposY+"%)";
	}
}