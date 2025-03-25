package com.jang.book.mapper;

import java.util.List;

import org.apache.ibatis.annotations.Mapper;

import com.jang.book.model.Book;

@Mapper
public interface BookMapper {
	Book getBook(String title);
	
	List<Book> getBookList();
	
	Book getBook(int bno);
	int addBook(Book book);
	int updateBook(Book book);
	int deleteBook(int bno);
	int nn();
}
