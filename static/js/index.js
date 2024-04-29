window.HELP_IMPROVE_VIDEOJS = false;

$(document).ready(function() {
    // Check for click events on the navbar burger icon

    var options = {
			slidesToScroll: 1,
			slidesToShow: 1,
			loop: true,
			infinite: true,
			autoplay: true,
			autoplaySpeed: 5000,
    }

		// Initialize all div with carousel class
    var carousels = bulmaCarousel.attach('.carousel', options);
	
    bulmaSlider.attach();

	// Get the video element
	var video = document.getElementById('reg1');
	// Set the playback rate to 0.5
	video.playbackRate = 0.5;
	video = document.getElementById('reg2');
	video.playbackRate = 0.5;
	video = document.getElementById('reg3');
	video.playbackRate = 0.5;
	video = document.getElementById('reg4');
	video.playbackRate = 0.5;



})