/**
 * 三目運算
 */
package hello;
import java.util.Scanner;
public class Score {
	public static void main(String[] args) {
		Scanner input = new Scanner(System.in);
		System.out.print("請輸入一個成績");
		int s = input.nextInt();
		boolean score = s>=60?true:false;
		/**
		 * String score = s>=60?"及格":"不及格";
		 */
		System.out.println(score);
	}
}
