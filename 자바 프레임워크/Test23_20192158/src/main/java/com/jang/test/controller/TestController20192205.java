package com.jang.test.controller;

import org.springframework.stereotype.Controller;
import org.springframework.ui.Model;
import org.springframework.web.bind.annotation.RequestMapping;

@Controller
public class TestController20192205 {
	
	@RequestMapping("/test")
	public String TestMsg20192205(Model model) {
		
		model.addAttribute("msg20192205","(20192205:이상윤)의 스프링프레임워크 답안입니다.");
		return"test20192205";
	}

}
